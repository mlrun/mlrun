# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy
import enum
import http
import typing

import requests.adapters
import requests.auth
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton
import server.api.utils.projects.remotes.follower as project_follower
from mlrun.utils import logger


class Client(
    project_follower.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self) -> None:
        super().__init__()
        self._session = mlrun.utils.HTTPSessionWithRetry(verbose=True)
        self._api_url = mlrun.mlconf.nuclio_dashboard_url

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.common.schemas.Project
    ):
        logger.debug("Creating project in Nuclio", project=project)
        body = self._generate_request_body(project)
        self._post_project_to_nuclio(body)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
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
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
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
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        logger.debug(
            "Deleting project in Nuclio", name=name, deletion_strategy=deletion_strategy
        )
        body = self._generate_request_body(
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name=name)
            )
        )
        headers = {
            "x-nuclio-delete-project-strategy": deletion_strategy.to_nuclio_deletion_strategy(),
        }
        try:
            self._send_request_to_api(
                "DELETE",
                "projects",
                auth_info=auth_info,
                json=body,
                headers=headers,
            )
        except requests.HTTPError as exc:
            if exc.response.status_code != http.HTTPStatus.NOT_FOUND.value:
                raise
            logger.debug(
                "Project not found in Nuclio. Considering deletion as successful",
                name=name,
                deletion_strategy=deletion_strategy,
            )

    def get_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.Project:
        response = self._get_project_from_nuclio(name, auth_info)
        response_body = response.json()
        return self._transform_nuclio_project_to_schema(response_body)

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectsOutput:
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
        if names:
            raise NotImplementedError(
                "Filtering nuclio projects by names is currently not supported"
            )
        response = self._send_request_to_api("GET", "projects", auth_info=auth_info)
        response_body = response.json()
        projects = []
        for nuclio_project in response_body.values():
            projects.append(self._transform_nuclio_project_to_schema(nuclio_project))
        if format_ == mlrun.common.schemas.ProjectsFormat.full:
            return mlrun.common.schemas.ProjectsOutput(projects=projects)
        elif format_ == mlrun.common.schemas.ProjectsFormat.name_only:
            return mlrun.common.schemas.ProjectsOutput(
                projects=[project.metadata.name for project in projects]
            )
        else:
            raise NotImplementedError(
                f"Provided format is not supported. format={format_}"
            )

    def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        raise NotImplementedError("Listing project summaries is not supported")

    def get_project_summary(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.common.schemas.ProjectSummary:
        raise NotImplementedError("Get project summary is not supported")

    def get_dashboard_version(self) -> str:
        response = self._send_request_to_api("GET", "versions")
        response_body = response.json()
        return response_body["dashboard"]["label"]

    def _get_project_from_nuclio(
        self, name, auth_info: mlrun.common.schemas.AuthInfo = None
    ):
        return self._send_request_to_api("GET", f"projects/{name}", auth_info=auth_info)

    def _post_project_to_nuclio(
        self, body, auth_info: mlrun.common.schemas.AuthInfo = None
    ):
        return self._send_request_to_api(
            "POST", "projects", auth_info=auth_info, json=body
        )

    def _put_project_to_nuclio(
        self, body, auth_info: mlrun.common.schemas.AuthInfo = None
    ):
        self._send_request_to_api("PUT", "projects", auth_info=auth_info, json=body)

    def _send_request_to_api(
        self,
        method,
        path,
        auth_info: mlrun.common.schemas.AuthInfo = None,
        **kwargs,
    ):
        url = f"{self._api_url}/api/{path}"
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20

        # requests no longer supports header values to be enum (https://github.com/psf/requests/pull/6154)
        # convert to strings. Do the same for params for niceness
        for kwarg in ["headers", "params"]:
            dict_ = kwargs.get(kwarg, {})
            for key in dict_.keys():
                if isinstance(dict_[key], enum.Enum):
                    dict_[key] = dict_[key].value

        auth = None
        if auth_info:
            auth = auth_info.to_nuclio_auth_info().to_requests_auth()

        response = self._session.request(
            method,
            url,
            verify=mlrun.mlconf.httpdb.http.verify,
            auth=auth,
            **kwargs,
        )
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
    def _generate_request_body(project: mlrun.common.schemas.Project):
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
        return mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=nuclio_project["metadata"]["name"],
                labels=nuclio_project["metadata"].get("labels"),
                annotations=nuclio_project["metadata"].get("annotations"),
            ),
            spec=mlrun.common.schemas.ProjectSpec(
                description=nuclio_project["spec"].get("description")
            ),
        )
