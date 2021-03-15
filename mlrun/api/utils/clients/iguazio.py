import copy
import typing

import requests.adapters
import urllib3

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.member
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Client(metaclass=mlrun.utils.singleton.Singleton,):
    def __init__(self) -> None:
        super().__init__()
        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.util.retry.Retry(total=3, backoff_factor=1)
        )
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._api_url = mlrun.mlconf.httpdb.iguazio_api_url

    def get_grafana_service_url_if_exists(
        self, session_cookie: str
    ) -> typing.Optional[str]:
        logger.debug("Getting grafana service url from Zebo")
        response = self._send_request_to_api(
            "GET", "app_services_manifests", session_cookie
        )
        response_body = response.json()
        if len(response_body.get("data", [{}])) > 1:
            raise mlrun.errors.MLRunInternalServerError(
                "Found unexpected number of app services manifests"
            )
        app_services = (
            response_body.get("data", [{}])[0]
            .get("attributes", {})
            .get("app_services", [])
        )
        for app_service in app_services:
            if (
                app_service.get("spec", {}).get("kind") == "grafana"
                and app_service.get("status", {}).get("state") == "ready"
                and len(app_service.get("status", {}).get("urls", [])) > 0
            ):
                # heuristically picking the first one
                return app_service.get("status", {}).get("urls")[0]["url"]
        return None

    def _send_request_to_api(self, method, path, session_cookie=None, **kwargs):
        url = f"{self._api_url}/api/{path}"
        if session_cookie:
            cookies = kwargs.get("cookies", {})
            if "session" in cookies and cookies["session"] != session_cookie:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Session cookie already set"
                )
            cookies["session"] = session_cookie
            kwargs["cookies"] = cookies
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        response = self._session.request(method, url, verify=False, **kwargs)
        if not response.ok:
            log_kwargs = copy.deepcopy(kwargs)
            log_kwargs.update({"method": method, "path": path})
            if response.content:
                try:
                    data = response.json()
                    ctx = data.get("meta", {}).get("ctx")
                    errors = data.get("errors", [])
                except Exception:
                    pass
                else:
                    log_kwargs.update({"ctx": ctx, "errors": errors})
            logger.warning("Request to iguazio failed", **log_kwargs)
            mlrun.errors.raise_for_status(response)
        return response
