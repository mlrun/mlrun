import copy
import typing
import http

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
        self._api_url = mlrun.mlconf.iguazio_api_url

    def try_get_grafana_service_url(self, session_cookie: str) -> typing.Optional[str]:
        """
        Try to find a ready grafana app service, and return its URL
        If nothing found, returns None
        """
        logger.debug("Getting grafana service url from Iguazio")
        response = self._send_request_to_api(
            "GET", "app_services_manifests", session_cookie
        )
        response_body = response.json()
        for app_services_manifest in response_body.get("data", []):
            for app_service in app_services_manifest.get("attributes", {}).get(
                "app_services", []
            ):
                if (
                    app_service.get("spec", {}).get("kind") == "grafana"
                    and app_service.get("status", {}).get("state") == "ready"
                    and len(app_service.get("status", {}).get("urls", [])) > 0
                ):
                    url_kind_to_url = {}
                    for url in app_service["status"]["urls"]:
                        url_kind_to_url[url["kind"]] = url["url"]
                    # precedence for https
                    for kind in ["https", "http"]:
                        if kind in url_kind_to_url:
                            return url_kind_to_url[kind]
        return None

    def create_project(self, session_cookie: str, project: mlrun.api.schemas.Project) -> mlrun.api.schemas.Project:
        logger.debug("Creating project in Iguazio", project=project)
        body = self._generate_request_body(project)
        return self._post_project_to_iguazio(session_cookie, body)

    def store_project(
        self,
        session_cookie: str,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        logger.debug("Storing project in Iguazio", name=name, project=project)
        body = self._generate_request_body(project)
        try:
            self._get_project_from_iguazio(name)
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            self._post_project_to_iguazio(session_cookie, body)
        else:
            self._put_project_to_iguazio(session_cookie, body)

    def delete_project(
        self,
        session_cookie: str,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
    ):
        logger.debug(
            "Deleting project in Iguazio", name=name, deletion_strategy=deletion_strategy
        )
        body = self._generate_request_body(
            mlrun.api.schemas.Project(
                metadata=mlrun.api.schemas.ProjectMetadata(name=name)
            )
        )
        # TODO: verify header name
        headers = {
            "x-iguazio-delete-project-strategy": deletion_strategy.to_nuclio_deletion_strategy(),
        }
        try:
            self._send_request_to_api("DELETE", "projects", session_cookie, json=body, headers=headers)
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            logger.debug(
                "Project not found in Iguazio. Considering deletion as successful",
                name=name,
                deletion_strategy=deletion_strategy,
            )

    def list_projects(
        self,
        session_cookie: str,
    ) -> typing.List[mlrun.api.schemas.Project]:
        response = self._send_request_to_api("GET", "projects", session_cookie)
        response_body = response.json()
        projects = []
        for iguazio_project in response_body.values():
            projects.append(self._transform_iguazio_project_to_schema(iguazio_project))
        return projects

    def _post_project_to_iguazio(self, session_cookie: str, body: dict) -> mlrun.api.schemas.Project:
        response = self._send_request_to_api("POST", "projects", session_cookie, json=body)
        return self._transform_iguazio_project_to_schema(response)

    def _put_project_to_iguazio(self, session_cookie: str, body: dict):
        response = self._send_request_to_api("PUT", "projects", session_cookie, json=body)
        return self._transform_iguazio_project_to_schema(response)

    def _get_project_from_iguazio(self, name):
        return self._send_request_to_api("GET", f"projects/{name}")

    def _send_request_to_api(self, method, path, session_cookie=None, **kwargs):
        url = f"{self._api_url}/api/{path}"
        if session_cookie:
            cookies = kwargs.get("cookies", {})
            # in case some dev using this function for some reason setting cookies manually through kwargs + have a
            # cookie with "session" key there + filling the session cookie - explode
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

    @staticmethod
    def _generate_request_body(project: mlrun.api.schemas.Project):
        # TODO: when I have stable schema
        body = {}
        return body

    @staticmethod
    def _transform_iguazio_project_to_schema(iguazio_project):
        # TODO: when I have stable schema
        return mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                # name=nuclio_project["metadata"]["name"],
                # labels=nuclio_project["metadata"].get("labels"),
                # annotations=nuclio_project["metadata"].get("annotations"),
            ),
            spec=mlrun.api.schemas.ProjectSpec(
                # description=nuclio_project["spec"].get("description")
            ),
        )
