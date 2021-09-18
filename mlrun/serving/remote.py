import json

import requests
import storey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import mlrun

http_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)


class RemoteStep(storey.SendToHttp):
    """class for calling remote endpoints
    """

    def __init__(
        self,
        url: str,
        subpath: str = None,
        method: str = None,
        headers: dict = None,
        url_expression: str = None,
        return_json: bool = True,
        **kwargs,
    ):
        """class for calling remote endpoints

        sync and async graph step implementation for request/resp to remote service (class shortcut = "$remote")
        url can be an http(s) url (e.g. "https://myservice/path") or an mlrun function uri ([project/]name).
        alternatively the url_expression can be specified to build the url from the event (e.g. "event['url']").

        example pipeline::

            flow = function.set_topology("flow", engine="async")
            flow.to(name="step1", handler="func1")\
                .to(RemoteStep(name="remote_echo", url="https://myservice/path", method="POST"))\
                .to(name="laststep", handler="func2").respond()


        :param url:     http(s) url or function [project/]name to call
        :param subpath: path (which follows the url), use `$path` to use the event.path
        :param method:  HTTP method (GET, POST, ..), default to POST
        :param headers: dictionary with http header values
        :param url_expression: an expression for getting the url from the event, e.g. "event['url']"
        :param return_json: indicate the returned value is json, and convert it to a py object
        """
        if url and url_expression:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot set both url and url_expression"
            )
        self.url = url
        self.url_expression = url_expression
        self.headers = headers
        self.method = method
        self.return_json = return_json
        self.subpath = subpath or ""
        super().__init__(None, None, **kwargs)

        self._append_event_path = False
        self._endpoint = ""
        self._session = None
        self._url_function_handler = None

    def post_init(self, mode="sync"):
        if self.url_expression:
            # init lambda function for calculating url from event
            self._url_function_handler = eval(
                "lambda event: " + self.url_expression, {}, {}
            )
        else:
            self._append_event_path = self.subpath == "$path"
            self._endpoint = self.context.get_remote_endpoint(self.url).strip("/")
            if self.subpath and not self._append_event_path:
                self._endpoint = self._endpoint + "/" + self.subpath.lstrip("/")

    async def _process_event(self, event):
        # async implementation (with storey)
        method, url, headers, body = self._generate_request(event)
        return await self._client_session.request(
            method, url, headers=headers, data=body, ssl=False
        )

    async def _handle_completed(self, event, response):
        response_body = await response.read()
        body = self._get_data(response_body, response.headers)

        if body is not None:
            new_event = self._user_fn_output_to_event(event, body)
            await self._do_downstream(new_event)

    def do_event(self, event):
        # sync implementation (without storey)
        if not self._session:
            self._session = requests.Session()
            self._session.mount("http://", http_adapter)
            self._session.mount("https://", http_adapter)

        method, url, headers, body = self._generate_request(event)
        try:
            resp = self._session.request(
                method, url, verify=False, headers=headers, data=body
            )
        except OSError as err:
            raise OSError(f"error: cannot invoke url: {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad http response {resp.text}")

        event.body = self._get_data(resp.content, resp.headers)
        return event

    def _generate_request(self, event):
        method = self.method or event.method or "POST"
        headers = self.headers or event.headers or {}

        body = None
        if method != "GET" and event.body is not None:
            if isinstance(event.body, (str, bytes)):
                body = event.body
            else:
                body = json.dumps(event.body)
                headers["Content-Type"] = "application/json"

        if self._url_function_handler:
            url = self._url_function_handler(event.body)
        else:
            url = self._endpoint
            if self._append_event_path:
                url = url + "/" + event.path.lstrip("/")
        return method, url, headers, body

    def _get_data(self, data, headers):
        if (
            self.return_json
            or headers.get("content-type", "").lower() == "application/json"
        ) and isinstance(data, (str, bytes)):
            data = json.loads(data)
        return data

    def to_dict(self):
        args = {
            key: getattr(self, key)
            for key in [
                "url",
                "subpath",
                "method",
                "headers",
                "return_json",
                "url_expression",
            ]
        }
        return {
            "class_name": f"{__name__}.{self.__class__.__name__}",
            "name": self.name or self.__class__.__name__,
            "class_args": args,
        }
