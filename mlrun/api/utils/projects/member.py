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

import mlrun.api.crud
import mlrun.api.db.session
import mlrun.api.utils.clients.log_collector
import mlrun.common.schemas
import mlrun.utils.singleton
from mlrun.utils import logger


class Member(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
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
        project_names = self.list_projects(
            db_session,
            format_=mlrun.common.schemas.ProjectsFormat.name_only,
            leader_session=auth_info.session,
        )
        if name not in project_names.projects:
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
    ) -> typing.Tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
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
    ) -> typing.Tuple[typing.Optional[mlrun.common.schemas.Project], bool]:
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
    ) -> typing.Tuple[mlrun.common.schemas.Project, bool]:
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
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        leader_session: typing.Optional[str] = None,
    ) -> mlrun.common.schemas.Project:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: typing.List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[typing.List[str]] = None,
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
        labels: typing.List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        projects_role: typing.Optional[mlrun.common.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        names: typing.Optional[typing.List[str]] = None,
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
            await self._stop_logs_for_project(project_name)
            await self._delete_project_logs(project_name)

    @staticmethod
    async def _stop_logs_for_project(
        project_name: str,
    ) -> None:

        logger.debug("Stopping logs for project", project=project_name)

        try:
            log_collector_client = (
                mlrun.api.utils.clients.log_collector.LogCollectorClient()
            )
            await log_collector_client.stop_logs(
                project=project_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed stopping logs for project's runs. Ignoring",
                exc=mlrun.errors.err_to_str(exc),
                project=project_name,
            )

        logger.debug(
            "Successfully stopped logs for project's runs", project=project_name
        )

    @staticmethod
    async def _delete_project_logs(
        project_name: str,
    ) -> None:

        logger.debug("Deleting logs for project", project=project_name)

        try:
            log_collector_client = (
                mlrun.api.utils.clients.log_collector.LogCollectorClient()
            )
            await log_collector_client.delete_logs(
                project=project_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed deleting project logs via the log collector. Falling back to deleting logs explicitly",
                exc=mlrun.errors.err_to_str(exc),
                project=project_name,
            )

            # fallback to deleting logs explicitly if the project logs deletion failed
            mlrun.api.crud.Logs().delete_logs(project_name)

        logger.debug("Successfully deleted project logs", project=project_name)
