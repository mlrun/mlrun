import json

import requests
import storey
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

http_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)


class RemoteState(storey.SendToHttp):
    """class for calling remote endpoints
    """

    def __init__(
        self, url, subpath=None, method=None, headers=None, return_json=True, **kwargs
    ):
        """class for calling remote endpoints

        :param url:     http(s) url or function [project/]name to call
        :param subpath: path (which follows the url), use `$path` to use the event.path
        :param method:  HTTP method (GET, POST, ..)
        :param headers: dictionary with http header values
        :param return_json: indicate the returned value is json, and convert it to a py object
        """
        self.url = url
        self.headers = headers
        self.method = method
        self.return_json = return_json
        self.subpath = subpath or ""
        super().__init__(None, None, **kwargs)

        self._append_event_path = False
        self._endpoint = ""
        self._session = None

    def post_init(self, mode="sync"):
        self._append_event_path = self.subpath == "$path"
        self._endpoint = self.context.get_remote_endpoint(self.url).strip("/")
        if self.subpath and not self._append_event_path:
            self._endpoint = self._endpoint + "/" + self.subpath.lstrip("/")

    async def _process_event(self, event):
        method, url, headers, body = self._gen_request(event)
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
        if not self._session:
            self._session = requests.Session()
            self._session.mount("http://", http_adapter)
            self._session.mount("https://", http_adapter)

        method, url, headers, body = self._gen_request(event)
        try:
            resp = self._session.request(
                method, url, verify=False, headers=headers, data=body
            )
        except OSError as err:
            raise OSError(f"error: cannot run function at url {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.text}")

        event.body = self._get_data(resp.content, resp.headers)
        return event

    def _gen_request(self, event):
        method = self.method or event.method or "POST"
        headers = self.headers or event.headers or {}

        body = None
        if method != "GET" and event.body is not None:
            if isinstance(event.body, (str, bytes)):
                body = event.body
            else:
                body = json.dumps(event.body)
                headers["Content-Type"] = "application/json"

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
            for key in ["url", "subpath", "method", "headers", "return_json"]
        }
        return {
            "class_name": "$remote",
            "name": self.name or "RemoteStep",
            "class_args": args,
        }
