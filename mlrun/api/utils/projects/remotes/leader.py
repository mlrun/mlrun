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
import datetime
import typing

import mlrun.common.schemas


class Member(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        session: str,
        updated_after: typing.Optional[datetime.datetime] = None,
    ) -> typing.Tuple[
        typing.List[mlrun.common.schemas.Project], typing.Optional[datetime.datetime]
    ]:
        pass

    @abc.abstractmethod
    def get_project(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.Project:
        pass

    @abc.abstractmethod
    def format_as_leader_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.IguazioProject:
        pass

    @abc.abstractmethod
    def get_project_owner(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        pass
