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
import abc
import typing

import sqlalchemy.orm

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.k8s_utils
import mlrun.utils.singleton
import server.api.crud
import server.api.utils.clients.log_collector


class Member(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    def ensure_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        wait_for_completion: bool = True,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        try:
            project = self.get_project(
                db_session,
                format_=mlrun.common.formatters.ProjectFormat.name_only,
                leader_session=auth_info.session,
                from_leader=False,
                name=name,
            )
        except mlrun.errors.MLRunNotFoundError:
            project = None

        # for custom description and for sanity check
        if not project:
            raise mlrun.errors.MLRunNotFoundError(f"Project {name} does not exist")

    @abc.abstractmethod
    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
        commit_before_get: bool = False,
    ) -> tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        pass

    @abc.abstractmethod
    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
        pass

    @abc.abstractmethod
    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> tuple[mlrun.common.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
        background_task_name: str = None,
        model_monitoring_access_key: str = None,
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
        from_leader: bool = False,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
    ) -> mlrun.common.schemas.ProjectOutput:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abc.abstractmethod
    async def get_project_summary(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.common.schemas.ProjectSummary:
        pass

    @abc.abstractmethod
    async def list_project_summaries(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        pass

    @abc.abstractmethod
    def get_project_owner(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        pass

    async def post_delete_project(
        self,
        project_name: str,
    ):
        if (
            mlrun.mlconf.log_collector.mode
            != mlrun.common.schemas.LogsCollectorMode.legacy
        ):
            await server.api.crud.Logs().stop_logs_for_project(project_name)
            await server.api.crud.Logs().delete_project_logs(project_name)

    def _validate_project(self, project: mlrun.common.schemas.Project):
        mlrun.projects.ProjectMetadata.validate_project_name(project.metadata.name)
        mlrun.projects.ProjectMetadata.validate_project_labels(project.metadata.labels)
        mlrun.k8s_utils.validate_node_selectors(
            project.spec.default_function_node_selector
        )
