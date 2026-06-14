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

import abc
import datetime

import sqlalchemy.orm

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.k8s_utils
import mlrun.utils.singleton

import framework.db.sqldb.models
import framework.utils.auth.verifier
import framework.utils.project_formats
import services.api.crud


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
            # Using minimal format to access spec.owner for OPA cache population
            # while avoiding the overhead of fetching large fields (functions, workflows, artifacts)
            project = self.get_project(
                db_session,
                format_=framework.utils.project_formats.ProjectFormatCustomSelection(
                    [
                        framework.utils.project_formats.ProjectFormatCustom.name,
                        framework.utils.project_formats.ProjectFormatCustom.owner,
                    ]
                ),
                auth_info=auth_info,
                from_leader=False,
                name=name,
            )
        except mlrun.errors.MLRunNotFoundError:
            project = None

        # for custom description and for sanity check
        if not project:
            raise mlrun.errors.MLRunNotFoundError(f"Project {name} does not exist")

        # Populate the OPA owner cache if the requesting user is the project owner.
        # This mitigates the OPA manifest propagation race condition on multi-pod deployments:
        # when a request is routed to a pod that hasn't received the OPA manifest yet,
        # the cache allows the owner to proceed without waiting for OPA propagation.
        verifier = framework.utils.auth.verifier.AuthVerifier()
        if verifier.is_project_owner(auth_info, project):
            verifier.add_allowed_project_for_owner(name, auth_info)

    @abc.abstractmethod
    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
        commit_before_get: bool = False,
    ) -> tuple[mlrun.common.schemas.Project | None, bool]:
        pass

    @abc.abstractmethod
    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> tuple[mlrun.common.schemas.Project | None, bool]:
        pass

    @abc.abstractmethod
    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> tuple[mlrun.common.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
        background_task_name: str | None = None,
        model_monitoring_access_key: str | None = None,
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        from_leader: bool = False,
        format_: framework.utils.project_formats.ProjectFormatType = mlrun.common.formatters.ProjectFormat.full,
    ) -> mlrun.common.schemas.ProjectOutput:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        owner: str | None = None,
        format_: framework.utils.project_formats.ProjectFormatType = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
        updated_after: datetime.datetime | None = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abc.abstractmethod
    async def get_project_summary(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectSummary:
        pass

    @abc.abstractmethod
    async def list_project_summaries(
        self,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        owner: str | None = None,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        pass

    @abc.abstractmethod
    def get_project_owner(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
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
            await services.api.crud.Logs().stop_logs_for_project(project_name)
            await services.api.crud.Logs().delete_project_logs(project_name)

    def _validate_project(self, project: mlrun.common.schemas.Project):
        mlrun.projects.ProjectMetadata.validate_project_name(project.metadata.name)
        mlrun.projects.ProjectMetadata.validate_project_labels(project.metadata.labels)
        self._validate_project_field_length("source", project.spec.source)
        self._validate_project_field_length("description", project.spec.description)
        self._validate_project_field_length("owner", project.spec.owner)
        mlrun.k8s_utils.validate_node_selectors(
            project.spec.default_function_node_selector
        )

    @staticmethod
    def _validate_project_field_length(field_name: str, value: str | None) -> None:
        """Validate a project text field against the backing DB column's max length.

        The limit is read from the ``Project`` model column itself, so it stays in sync
        with the schema instead of duplicating the size. A longer value would otherwise
        reach the DB and fail with an opaque 500; reject it up front with a clear 400.

        :param field_name: name of the project field (must match the model column name)
        :param value:      the value to validate (``None``/empty is a no-op)
        :raises mlrun.errors.MLRunInvalidArgumentError: if the value exceeds the column limit
        """
        if not value:
            return
        column = getattr(framework.db.sqldb.models.Project, field_name)
        max_length = column.type.max_length
        if len(value) > max_length:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Project '{field_name}' length ({len(value)}) exceeds the maximum allowed "
                f"length of {max_length} characters"
            )
