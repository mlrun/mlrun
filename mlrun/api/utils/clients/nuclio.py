import copy
import http
import typing

import requests.adapters
import sqlalchemy.orm
import urllib3

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.member
import mlrun.errors
import mlrun.utils.singleton
from mlrun.utils import logger


class Client(
    mlrun.api.utils.projects.remotes.member.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self) -> None:
        super().__init__()
        http_adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.util.retry.Retry(total=3, backoff_factor=1)
        )
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._api_url = mlrun.config.config.nuclio_dashboard_url

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.Project
    ):
        logger.debug("Creating project in Nuclio", project=project)
        body = self._generate_request_body(project)
        self._post_project_to_nuclio(body)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        logger.debug("Storing project in Nuclio", name=name, project=project)
        body = self._generate_request_body(project)
        try:
            self._get_project_from_nuclio(name)
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            self._post_project_to_nuclio(body)
        else:
            self._put_project_to_nuclio(body)

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ):
        logger.debug(
            "Patching project in Nuclio",
            name=name,
            project=project,
            patch_mode=patch_mode,
        )
        response = self._get_project_from_nuclio(name)
        response_body = response.json()
        if project.get("metadata", {}).get("labels") is not None:
            response_body.setdefault("metadata", {}).setdefault("labels", {}).update(
                project["metadata"]["labels"]
            )
        if project.get("metadata", {}).get("annotations") is not None:
            response_body.setdefault("metadata", {}).setdefault(
                "annotations", {}
            ).update(project["metadata"]["annotations"])
        if project.get("spec", {}).get("description") is not None:
            response_body.setdefault("spec", {})["description"] = project["spec"][
                "description"
            ]
        self._put_project_to_nuclio(response_body)

    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
    ):
        logger.debug(
            "Deleting project in Nuclio", name=name, deletion_strategy=deletion_strategy
        )
        body = self._generate_request_body(
            mlrun.api.schemas.Project(
                metadata=mlrun.api.schemas.ProjectMetadata(name=name)
            )
        )
        headers = {
            "x-nuclio-delete-project-strategy": deletion_strategy.to_nuclio_deletion_strategy(),
        }
        try:
            self._send_request_to_api("DELETE", "projects", json=body, headers=headers)
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            logger.debug(
                "Project not found in Nuclio. Considering deletion as successful",
                name=name,
                deletion_strategy=deletion_strategy,
            )

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        response = self._get_project_from_nuclio(name)
        response_body = response.json()
        return self._transform_nuclio_project_to_schema(response_body)

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        if owner:
            raise NotImplementedError(
                "Listing nuclio projects by owner is currently not supported"
            )
        if labels:
            raise NotImplementedError(
                "Filtering nuclio projects by labels is currently not supported"
            )
        if state:
            raise NotImplementedError(
                "Filtering nuclio projects by state is currently not supported"
            )
        response = self._send_request_to_api("GET", "projects")
        response_body = response.json()
        projects = []
        for nuclio_project in response_body.values():
            projects.append(self._transform_nuclio_project_to_schema(nuclio_project))
        if format_ == mlrun.api.schemas.Format.full:
            return mlrun.api.schemas.ProjectsOutput(projects=projects)
        elif format_ == mlrun.api.schemas.Format.name_only:
            return mlrun.api.schemas.ProjectsOutput(
                projects=[project.metadata.name for project in projects]
            )
        else:
            raise NotImplementedError(
                f"Provided format is not supported. format={format_}"
            )

    def _get_project_from_nuclio(self, name):
        return self._send_request_to_api("GET", f"projects/{name}")

    def _post_project_to_nuclio(self, body):
        return self._send_request_to_api("POST", "projects", json=body)

    def _put_project_to_nuclio(self, body):
        self._send_request_to_api("PUT", "projects", json=body)

    def _send_request_to_api(self, method, path, **kwargs):
        url = f"{self._api_url}/api/{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
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
            logger.warning("Request to nuclio failed", **log_kwargs)
            mlrun.errors.raise_for_status(response)
        return response

    @staticmethod
    def _generate_request_body(project: mlrun.api.schemas.Project):
        body = {
            "metadata": {"name": project.metadata.name},
        }
        if project.metadata.labels:
            body["metadata"]["labels"] = project.metadata.labels
        if project.metadata.annotations:
            body["metadata"]["annotations"] = project.metadata.annotations
        if project.spec.description:
            body["spec"] = {"description": project.spec.description}
        return body

    @staticmethod
    def _transform_nuclio_project_to_schema(nuclio_project):
        return mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=nuclio_project["metadata"]["name"],
                labels=nuclio_project["metadata"].get("labels"),
                annotations=nuclio_project["metadata"].get("annotations"),
            ),
            spec=mlrun.api.schemas.ProjectSpec(
                description=nuclio_project["spec"].get("description")
            ),
        )
