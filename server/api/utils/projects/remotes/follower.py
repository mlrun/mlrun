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

import mlrun.common.schemas


class Member(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.common.schemas.Project
    ):
        pass

    @abc.abstractmethod
    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        pass

    @abc.abstractmethod
    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ):
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        pass

    @abc.abstractmethod
    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.common.schemas.Project:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.schemas.ProjectsFormat = mlrun.common.schemas.ProjectsFormat.full,
        labels: typing.List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abc.abstractmethod
    def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        labels: typing.List[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        pass

    @abc.abstractmethod
    def get_project_summary(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.common.schemas.ProjectSummary:
        pass
