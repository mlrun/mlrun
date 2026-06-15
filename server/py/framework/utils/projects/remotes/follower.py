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
import uuid

import sqlalchemy.orm

import mlrun.common.formatters
import mlrun.common.schemas

import framework.utils.project_formats


class Member(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self,
        session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        pass

    @abc.abstractmethod
    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        pass

    @abc.abstractmethod
    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        pass

    @abc.abstractmethod
    def get_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.Project:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
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
    def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        owner: str | None = None,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        pass

    @abc.abstractmethod
    def get_project_summary(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectSummary:
        pass

    # ----- 2PC follower hooks ----------------------------------------------
    # Invoked by the 2PC orchestrator on each remote follower as part of
    # parallel fan-out. They take only data persisted on the project row, so
    # request-driven and reconciliation-driven invocations are equivalent —
    # no per-request session, no per-request auth_info. Concrete followers
    # authenticate using their own service-account credentials.

    @abc.abstractmethod
    def prepare_create_project(
        self,
        project: mlrun.common.schemas.Project,
        op_id: uuid.UUID,
    ) -> None:
        pass

    @abc.abstractmethod
    def commit_create_project(
        self,
        name: str,
        op_id: uuid.UUID,
    ) -> None:
        pass

    @abc.abstractmethod
    def prepare_delete_project(
        self,
        name: str,
        op_id: uuid.UUID,
    ) -> None:
        pass

    @abc.abstractmethod
    def commit_delete_project(
        self,
        name: str,
        op_id: uuid.UUID,
    ) -> None:
        pass

    @abc.abstractmethod
    def update_project_follower(
        self,
        name: str,
        project: mlrun.common.schemas.Project,
        op_id: uuid.UUID,
    ) -> None:
        pass
