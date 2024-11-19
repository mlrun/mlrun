# Copyright 2024 Iguazio
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

import datetime
import typing

import mlrun.common.schemas
import mlrun.common.types

from .base import ObjectFormat


class ProjectFormat(ObjectFormat, mlrun.common.types.StrEnum):
    full = "full"
    name_only = "name_only"
    # minimal format removes large fields from the response (e.g. functions, workflows, artifacts)
    # and is used for faster response times (in the UI)
    minimal = "minimal"
    # internal - allowed only in follower mode, only for the leader for upgrade purposes
    leader = "leader"

    name_and_creation_time = "name_and_creation_time"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        def _name_only(project: mlrun.common.schemas.Project) -> str:
            return project.metadata.name

        def _name_and_creation_time(
            project: mlrun.common.schemas.Project,
        ) -> tuple[str, datetime.datetime]:
            return project.metadata.name, project.metadata.created

        def _minimal(
            project: mlrun.common.schemas.Project,
        ) -> mlrun.common.schemas.Project:
            project.spec.functions = None
            project.spec.workflows = None
            project.spec.artifacts = None
            return project

        return {
            ProjectFormat.full: None,
            ProjectFormat.name_only: _name_only,
            ProjectFormat.minimal: _minimal,
            ProjectFormat.leader: None,
            ProjectFormat.name_and_creation_time: _name_and_creation_time,
        }[_format]
