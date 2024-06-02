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
import typing

import mergedeep
import sqlalchemy.orm

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.errors
import server.api.utils.projects.remotes.follower as project_follower


class Member(project_follower.Member):
    def __init__(self) -> None:
        super().__init__()
        self._projects: dict[str, mlrun.common.schemas.Project] = {}

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.common.schemas.Project
    ):
        if project.metadata.name in self._projects:
            raise mlrun.errors.MLRunConflictError("Project already exists")
        # deep copy so we won't accidentally get changes from tests
        self._projects[project.metadata.name] = project.copy(deep=True)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        # deep copy so we won't accidentally get changes from tests
        self._projects[name] = project.copy(deep=True)

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ):
        existing_project_dict = self._projects[name].dict()
        strategy = patch_mode.to_mergedeep_strategy()
        mergedeep.merge(existing_project_dict, project, strategy=strategy)
        self._projects[name] = mlrun.common.schemas.Project(**existing_project_dict)

    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        if name in self._projects:
            del self._projects[name]

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.common.schemas.Project:
        # deep copy so we won't accidentally get changes from tests
        return self._projects[name].copy(deep=True)

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        if owner or labels or state:
            raise NotImplementedError(
                "Filtering by owner, labels or state is not supported"
            )
        projects = list(self._projects.values())
        # deep copy so we won't accidentally get changes from tests
        projects = [project.copy(deep=True) for project in projects]
        if names:
            projects = [
                project
                for project_name, project in self._projects.items()
                if project_name in names
            ]

        return mlrun.common.schemas.ProjectsOutput(
            projects=[
                mlrun.common.formatters.ProjectFormat.format_obj(
                    project,
                    format_,
                    exclude_formats=[mlrun.common.formatters.ProjectFormat.leader],
                )
                for project in projects
            ]
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
