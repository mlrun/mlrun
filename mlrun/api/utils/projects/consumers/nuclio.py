import sqlalchemy.orm
import requests.adapters

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import requests.packages.urllib3.util.retry
from mlrun.utils import logger


class Consumer(mlrun.api.utils.projects.consumers.base.Consumer):
    def __init__(self) -> None:
        super().__init__()
        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.packages.urllib3.util.retry.Retry(
                total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
            )
        )
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._api_url = mlrun.config.config.nuclio_dashboard_url

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        body = {
            "metadata": {"name": project.name,},
            "spec": {"description": project.description},
        }
        self._send_request_to_api("POST", "projects", json=body)

    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        body = {
            "metadata": {"name": name,},
            "spec": {"description": project.description},
        }
        self._send_request_to_api("PUT", f"projects/{name}", json=body)

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        self._send_request_to_api("DELETE", f"projects/{name}")

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectOut:
        response = self._send_request_to_api("GET", f"projects/{name}")
        response_body = response.json()
        project = self._transform_nuclio_project_to_schema(response_body)
        return mlrun.api.schemas.ProjectOut(project=project)

    def list_projects(
        self, session: sqlalchemy.orm.Session
    ) -> mlrun.api.schemas.ProjectsOutput:
        response = self._send_request_to_api("GET", "projects")
        response_body = response.json()
        projects = []
        for nuclio_project in response_body.values():
            projects.append(self._transform_nuclio_project_to_schema(nuclio_project))
        return mlrun.api.schemas.ProjectsOutput(projects=projects)

    def _send_request_to_api(
        self, method, path, params=None, body=None, json=None, headers=None, timeout=20
    ):
        url = f"{self._api_url}/api/{path}"
        kwargs = {
            key: value
            for key, value in (
                ("params", params),
                ("data", body),
                ("json", json),
                ("headers", headers),
            )
            if value is not None
        }
        response = self._session.request(
            method, url, timeout=timeout, verify=False, **kwargs
        )
        if not response.ok:
            log_kwargs = {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
            if response.content:
                try:
                    data = response.json()
                    error = data.get("error")
                    error_stack_trace = data.get("errorStackTrace")
                except Exception:
                    pass
                else:
                    log_kwargs.update(
                        {"error": error, "error_stack_trace": error_stack_trace,}
                    )
            logger.warning("Request to nuclio failed", **log_kwargs)
            response.raise_for_status()
        return response

    @staticmethod
    def _transform_nuclio_project_to_schema(nuclio_project):
        return mlrun.api.schemas.Project(
            name=nuclio_project["metadata"]["name"],
            description=nuclio_project["spec"].get("description"),
        )
