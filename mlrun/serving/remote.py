import asyncio
import json

import aiohttp
import requests
import storey
from requests.adapters import HTTPAdapter
from storey.flow import _ConcurrentJobExecution
from urllib3.util.retry import Retry

import mlrun
from mlrun.utils import logger

from .utils import (
    _extract_input_data,
    _update_result_body,
    event_id_key,
    event_path_key,
)

default_retries = 6
default_backoff_factor = 1


def get_http_adapter(retries, backoff_factor):
    if retries != 0:
        retry = Retry(
            total=retries or default_retries,
            backoff_factor=default_backoff_factor
            if backoff_factor is None
            else backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            method_whitelist=False,
        )
    else:
        retry = Retry(0, read=False)
    return HTTPAdapter(max_retries=retry)


class RemoteStep(storey.SendToHttp):
    """class for calling remote endpoints"""

    def __init__(
        self,
        url: str,
        subpath: str = None,
        method: str = None,
        headers: dict = None,
        url_expression: str = None,
        body_expression: str = None,
        return_json: bool = True,
        input_path: str = None,
        result_path: str = None,
        max_in_flight=None,
        retries=None,
        backoff_factor=None,
        timeout=None,
        **kwargs,
    ):
        """class for calling remote endpoints

        sync and async graph step implementation for request/resp to remote service (class shortcut = "$remote")
        url can be an http(s) url (e.g. "https://myservice/path") or an mlrun function uri ([project/]name).
        alternatively the url_expression can be specified to build the url from the event (e.g. "event['url']").

        example pipeline::

            flow = function.set_topology("flow", engine="async")
            flow.to(name="step1", handler="func1")
                .to(RemoteStep(name="remote_echo", url="https://myservice/path", method="POST"))
                .to(name="laststep", handler="func2").respond()


        :param url:     http(s) url or function [project/]name to call
        :param subpath: path (which follows the url), use `$path` to use the event.path
        :param method:  HTTP method (GET, POST, ..), default to POST
        :param headers: dictionary with http header values
        :param url_expression: an expression for getting the url from the event, e.g. "event['url']"
        :param body_expression: an expression for getting the request body from the event, e.g. "event['data']"
        :param return_json: indicate the returned value is json, and convert it to a py object
        :param input_path:  when specified selects the key/path in the event to use as body
                            this require that the event body will behave like a dict, example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means request body will be 7
        :param result_path: selects the key/path in the event to write the results to
                            this require that the event body will behave like a dict, example:
                            event: {"x": 5} , result_path="resp" means the returned response will be written
                            to event["y"] resulting in {"x": 5, "resp": <result>}
        :param retries:     number of retries (in exponential backoff)
        :param backoff_factor: A backoff factor in seconds to apply between attempts after the second try
        :param timeout:     How long to wait for the server to send data before giving up, float in seconds
        """
        # init retry args for storey
        retries = default_retries if retries is None else retries
        super().__init__(
            None,
            None,
            input_path=input_path,
            result_path=result_path,
            max_in_flight=max_in_flight,
            retries=retries,
            backoff_factor=backoff_factor,
            **kwargs,
        )
        self.url = url
        self.url_expression = url_expression
        self.body_expression = body_expression
        self.headers = headers
        self.method = method
        self.return_json = return_json
        self.subpath = subpath

        self.timeout = timeout

        self._append_event_path = False
        self._endpoint = ""
        self._session = None
        self._url_function_handler = None
        self._body_function_handler = None

    def post_init(self, mode="sync"):
        self._endpoint = self.url
        if self.url and self.context:
            self._endpoint = self.context.get_remote_endpoint(self.url).strip("/")
        if self.body_expression:
            # init lambda function for calculating url from event
            self._body_function_handler = eval(
                "lambda event: " + self.body_expression, {"context": self.context}, {}
            )
        if self.url_expression:
            # init lambda function for calculating url from event
            self._url_function_handler = eval(
                "lambda event: " + self.url_expression,
                {"endpoint": self._endpoint, "context": self.context},
                {},
            )
        elif self.subpath:
            self._append_event_path = self.subpath == "$path"
            if not self._append_event_path:
                self._endpoint = self._endpoint + "/" + self.subpath.lstrip("/")

    async def _process_event(self, event):
        # async implementation (with storey)
        body = self._get_event_or_body(event)
        method, url, headers, body = self._generate_request(event, body)
        kwargs = {}
        if self.timeout:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=self.timeout)
        try:
            resp = await self._client_session.request(
                method, url, headers=headers, data=body, ssl=False, **kwargs
            )
            if resp.status >= 500:
                text = await resp.text()
                raise RuntimeError(f"bad http response {resp.status}: {text}")
            return resp
        except asyncio.TimeoutError as exc:
            logger.error(f"http request to {url} timed out in RemoteStep {self.name}")
            raise exc

    async def _handle_completed(self, event, response):
        response_body = await response.read()
        if response.status >= 400:
            raise ValueError(
                f"For event {event}, RemoteStep {self.name} got an unexpected response "
                f"status {response.status}: {response_body}"
            )

        body = self._get_data(response_body, response.headers)

        new_event = self._user_fn_output_to_event(event, body)
        await self._do_downstream(new_event)

    def do_event(self, event):
        # sync implementation (without storey)
        if not self._session:
            self._session = requests.Session()
            http_adapter = get_http_adapter(self.retries, self.backoff_factor)
            self._session.mount("http://", http_adapter)
            self._session.mount("https://", http_adapter)

        body = _extract_input_data(self._input_path, event.body)
        method, url, headers, body = self._generate_request(event, body)
        try:
            resp = self._session.request(
                method,
                url,
                verify=False,
                headers=headers,
                data=body,
                timeout=self.timeout,
            )
        except requests.exceptions.ReadTimeout as err:
            raise requests.exceptions.ReadTimeout(
                f"http request to {url} timed out in RemoteStep {self.name}, {err}"
            )
        except OSError as err:
            raise OSError(f"cannot invoke url: {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad http response {resp.status_code}: {resp.text}")

        result = self._get_data(resp.content, resp.headers)
        event.body = _update_result_body(self._result_path, event.body, result)
        return event

    def _generate_request(self, event, body):
        method = self.method or event.method or "POST"
        headers = self.headers or {}

        if self._url_function_handler:
            url = self._url_function_handler(body)
        else:
            url = self._endpoint
            striped_path = event.path.lstrip("/")
            if self._append_event_path:
                url = url + "/" + striped_path
            if striped_path:
                headers[event_path_key] = event.path

        if event.id:
            headers[event_id_key] = event.id

        if method == "GET":
            body = None
        elif body is not None and not isinstance(body, (str, bytes)):
            if self._body_function_handler:
                body = self._body_function_handler(body)
            body = json.dumps(body)
            headers["Content-Type"] = "application/json"

        return method, url, headers, body

    def _get_data(self, data, headers):
        if (
            self.return_json
            or headers.get("content-type", "").lower() == "application/json"
        ) and isinstance(data, (str, bytes)):
            data = json.loads(data)
        return data


class BatchHttpRequests(_ConcurrentJobExecution):
    """class for calling remote endpoints in parallel"""

    def __init__(
        self,
        url: str = None,
        subpath: str = None,
        method: str = None,
        headers: dict = None,
        url_expression: str = None,
        body_expression: str = None,
        return_json: bool = True,
        input_path: str = None,
        result_path: str = None,
        retries=None,
        backoff_factor=None,
        timeout=None,
        **kwargs,
    ):
        """class for calling remote endpoints in parallel

        sync and async graph step implementation for request/resp to remote service (class shortcut = "$remote")
        url can be an http(s) url (e.g. "https://myservice/path") or an mlrun function uri ([project/]name).
        alternatively the url_expression can be specified to build the url from the event (e.g. "event['url']").

        example pipeline::

            function = mlrun.new_function("myfunc", kind="serving")
            flow = function.set_topology("flow", engine="async")
            flow.to(
                BatchHttpRequests(
                    url_expression="event['url']",
                    body_expression="event['data']",
                    method="POST",
                    input_path="req",
                    result_path="resp",
                )
            ).respond()

            server = function.to_mock_server()
            # request contains a list of elements, each with url and data
            request = [{"url": f"{base_url}/{i}", "data": i} for i in range(2)]
            resp = server.test(body={"req": request})


        :param url:     http(s) url or function [project/]name to call
        :param subpath: path (which follows the url)
        :param method:  HTTP method (GET, POST, ..), default to POST
        :param headers: dictionary with http header values
        :param url_expression: an expression for getting the url from the event, e.g. "event['url']"
        :param body_expression: an expression for getting the request body from the event, e.g. "event['data']"
        :param return_json: indicate the returned value is json, and convert it to a py object
        :param input_path:  when specified selects the key/path in the event to use as body
                            this require that the event body will behave like a dict, example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means request body will be 7
        :param result_path: selects the key/path in the event to write the results to
                            this require that the event body will behave like a dict, example:
                            event: {"x": 5} , result_path="resp" means the returned response will be written
                            to event["y"] resulting in {"x": 5, "resp": <result>}
        :param retries:     number of retries (in exponential backoff)
        :param backoff_factor: A backoff factor in seconds to apply between attempts after the second try
        :param timeout:     How long to wait for the server to send data before giving up, float in seconds
        """
        if url and url_expression:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot set both url and url_expression"
            )
        self.url = url
        self.url_expression = url_expression
        self.body_expression = body_expression
        self.headers = headers
        self.method = method
        self.return_json = return_json
        self.subpath = subpath
        super().__init__(input_path=input_path, result_path=result_path, **kwargs)

        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._append_event_path = False
        self._endpoint = ""
        self._session = None
        self._url_function_handler = None
        self._body_function_handler = None
        self._request_args = {}

    def _init(self):
        super()._init()
        self._client_session = None

    async def _lazy_init(self):
        connector = aiohttp.TCPConnector()
        self._client_session = aiohttp.ClientSession(connector=connector)

    async def _cleanup(self):
        await self._client_session.close()

    def post_init(self, mode="sync"):
        self._endpoint = self.url
        if self.url and self.context:
            self._endpoint = self.context.get_remote_endpoint(self.url).strip("/")
        if self.body_expression:
            # init lambda function for calculating url from event
            self._body_function_handler = eval(
                "lambda event: " + self.body_expression, {}, {}
            )
        if self.url_expression:
            # init lambda function for calculating url from event
            self._url_function_handler = eval(
                "lambda event: " + self.url_expression, {}, {"endpoint": self._endpoint}
            )
        elif self.subpath:
            self._endpoint = self._endpoint + "/" + self.subpath.lstrip("/")
        if self.timeout:
            self._request_args["timeout"] = aiohttp.ClientTimeout(total=self.timeout)

    async def _process_event(self, event):
        # async implementation (with storey)
        method = self.method or event.method or "POST"
        headers = self.headers or {}
        input_list = self._get_event_or_body(event)
        is_get = method == "GET"
        is_json = False

        body_list = []
        url_list = []
        for body in input_list:
            if self._url_function_handler:
                url_list.append(self._url_function_handler(body))
            else:
                url_list.append(self._endpoint)

            if is_get:
                body = None
            elif body is not None and not isinstance(body, (str, bytes)):
                if self._body_function_handler:
                    body = self._body_function_handler(body)
                body = json.dumps(body)
                is_json = True
            body_list.append(body)

        if is_json:
            headers["Content-Type"] = "application/json"

        responses = []
        for url, body in zip(url_list, body_list):
            responses.append(
                asyncio.ensure_future(
                    self._submit_with_retries(method, url, headers, body)
                )
            )
        return await asyncio.gather(*responses)

    async def _process_event_with_retries(self, event):
        return await self._process_event(event)

    async def _submit(self, method, url, headers, body):
        async with self._client_session.request(
            method, url, headers=headers, data=body, ssl=False, **self._request_args
        ) as future:
            if future.status >= 500:
                text = await future.text()
                raise RuntimeError(f"bad http response {future.status}: {text}")
            return await future.read(), future.headers

    async def _submit_with_retries(self, method, url, headers, body):
        times_attempted = 0
        max_attempts = (self.retries or default_retries) + 1
        while True:
            try:
                return await self._submit(method, url, headers, body)
            except Exception as ex:
                times_attempted += 1
                attempts_left = max_attempts - times_attempted
                if self.logger:
                    self.logger.warn(
                        f"{self.name} failed to process event ({attempts_left} retries left): {ex}"
                    )
                if attempts_left <= 0:
                    raise ex
                backoff_factor = (
                    default_backoff_factor
                    if self.backoff_factor is None
                    else self.backoff_factor
                )
                backoff_value = (backoff_factor) * (2 ** (times_attempted - 1))
                backoff_value = min(self._BACKOFF_MAX, backoff_value)
                if backoff_value >= 0:
                    await asyncio.sleep(backoff_value)

    async def _handle_completed(self, event, response):
        data = []
        for body, headers in response:
            data.append(self._get_data(body, headers))

        new_event = self._user_fn_output_to_event(event, data)
        await self._do_downstream(new_event)

    def _get_data(self, data, headers):
        if (
            self.return_json
            or headers.get("content-type", "").lower() == "application/json"
        ) and isinstance(data, (str, bytes)):
            data = json.loads(data)
        return data
