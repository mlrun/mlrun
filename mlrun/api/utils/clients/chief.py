import copy

import requests.adapters
import urllib3

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Client(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    """
    We chose chief-workers architecture to provide multi-instance API.
    By default, all API calls can access both the chief and workers.
    The key distinction is that some responsibilities, such as scheduling jobs, are exclusively performed by the chief.
    Instead of limiting the ui/client to only send requests to the chief, which would cause the entire scaling solution
    to be redundant.
    When one of the workers receives a request that the chief needs to execute or may have the knowledge of that piece
    of information, the worker will redirect the request to the chief.
    """

    def __init__(self) -> None:
        super().__init__()
        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.util.retry.Retry(total=3, backoff_factor=1),
            pool_maxsize=int(mlrun.mlconf.httpdb.max_workers),
        )
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._api_url = mlrun.mlconf.resolve_chief_api_url()

    def get_background_task(self, name) -> mlrun.api.schemas.BackgroundTask:
        response = self._send_request_to_api("GET", f"background-tasks/{name}")
        response_body = response.json()
        return mlrun.api.schemas.BackgroundTask(**response_body)

    def trigger_migrations(self) -> mlrun.api.schemas.BackgroundTask:
        response = self._send_request_to_api("POST", "operations/migrations")
        response_body = response.json()
        return mlrun.api.schemas.BackgroundTask(**response_body)

    def _send_request_to_api(self, method, path, **kwargs):
        url = f"{self._api_url}/api/{mlrun.mlconf.api_base_version}/{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        logger.debug("Sending request to chief", method=method, url=url, **kwargs)
        response = self._session.request(method, url, verify=False, **kwargs)
        if not response.ok:
            log_kwargs = copy.deepcopy(kwargs)
            log_kwargs.update({"method": method, "path": path})
            if response.content:
                try:
                    data = response.json()
                    error = data.get("error")
                    error_stack_trace = data.get("errorStackTrace")
                except Exception:
                    pass
                else:
                    log_kwargs.update(
                        {"error": error, "error_stack_trace": error_stack_trace}
                    )
            logger.warning("Request to chief failed", **log_kwargs)
            mlrun.errors.raise_for_status(response)
        logger.debug(
            "Request to chief succeeded",
            method=method,
            url=url,
            **kwargs,
            response=response.json(),
        )
        return response
