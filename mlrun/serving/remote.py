import json

import requests
import storey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import _extract_input_data, _update_result_body

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
        body_expression: str = None,
        return_json: bool = True,
        input_path: str = None,
        result_path: str = None,
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
        :param body_expression: an expression for getting the request body from the event, e.g. "event['data']"
        :param return_json: indicate the returned value is json, and convert it to a py object
        :param input_path:  when specified selects the key/path in the event to use as body
                            this require that the event body will behave like a dict, example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means request body will be 7
        :param result_path: selects the key/path in the event to write the results to
                            this require that the event body will behave like a dict, example:
                            event: {"x": 5} , result_path="resp" means the returned response will be written
                            to event["y"] resulting in {"x": 5, "resp": <result>}
        """
        self.url = url
        self.url_expression = url_expression
        self.body_expression = body_expression
        self.headers = headers
        self.method = method
        self.return_json = return_json
        self.subpath = subpath
        super().__init__(
            None, None, input_path=input_path, result_path=result_path, **kwargs
        )

        self._append_event_path = False
        self._endpoint = ""
        self._session = None
        self._url_function_handler = None
        self._body_function_handler = None
        self._full_event = False

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
                "lambda event: " + self.url_expression, {"endpoint": self._endpoint}, {}
            )
        elif self.subpath:
            self._append_event_path = self.subpath == "$path"
            if not self._append_event_path:
                self._endpoint = self._endpoint + "/" + self.subpath.lstrip("/")

    async def _process_event(self, event):
        # async implementation (with storey)
        body = self._get_event_or_body(event)
        method, url, headers, body = self._generate_request(event, body)
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

        body = _extract_input_data(self._input_path, event.body)
        method, url, headers, body = self._generate_request(event, body)
        try:
            resp = self._session.request(
                method, url, verify=False, headers=headers, data=body
            )
        except OSError as err:
            raise OSError(f"error: cannot invoke url: {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad http response {resp.text}")

        result = self._get_data(resp.content, resp.headers)
        event.body = _update_result_body(self._result_path, event.body, result)
        return event

    def _generate_request(self, event, body):
        method = self.method or event.method or "POST"
        headers = self.headers or event.headers or {}

        if self._url_function_handler:
            url = self._url_function_handler(body)
        else:
            url = self._endpoint
            if self._append_event_path:
                url = url + "/" + event.path.lstrip("/")

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
                "body_expression",
            ]
            if getattr(self, key) is not None
        }
        return {
            "class_name": f"{__name__}.{self.__class__.__name__}",
            "name": self.name or self.__class__.__name__,
            "class_args": args,
            "input_path": self._input_path,
            "result_path": self._result_path,
        }
