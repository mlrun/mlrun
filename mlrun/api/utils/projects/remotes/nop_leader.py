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
import datetime
import typing

import mlrun.api.utils.projects.remotes.leader as project_leader
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
import mlrun.errors


class Member(project_leader.Member):
    def __init__(self) -> None:
        super().__init__()
        self.db_session = None
        self.project_owner_access_key = ""
        self._project_role = mlrun.common.schemas.ProjectsRole.nop

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        wait_for_completion: bool = True,
    ) -> bool:
        self._update_state(project)
        (
            _,
            is_running_in_background,
        ) = mlrun.api.utils.singletons.project_member.get_project_member().create_project(
            self.db_session, project, self._project_role
        )
        return is_running_in_background

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        self._update_state(project)
        mlrun.api.utils.singletons.project_member.get_project_member().store_project(
            self.db_session, name, project, self._project_role
        )

    @staticmethod
    def _update_state(project: mlrun.common.schemas.Project):
        if (
            not project.status.state
            or project.status.state
            in mlrun.common.schemas.ProjectState.terminal_states()
        ):
            project.status.state = mlrun.common.schemas.ProjectState(
                project.spec.desired_state
            )

    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        return mlrun.api.utils.singletons.project_member.get_project_member().delete_project(
            self.db_session, name, deletion_strategy, self._project_role
        )

    def list_projects(
        self,
        session: str,
        updated_after: typing.Optional[datetime.datetime] = None,
    ) -> typing.Tuple[
        typing.List[mlrun.common.schemas.Project], typing.Optional[datetime.datetime]
    ]:
        return (
            mlrun.api.utils.singletons.project_member.get_project_member()
            .list_projects(self.db_session)
            .projects,
            datetime.datetime.utcnow(),
        )

    def get_project(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.Project:
        return (
            mlrun.api.utils.singletons.project_member.get_project_member().get_project(
                self.db_session, name
            )
        )

    def format_as_leader_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.IguazioProject:
        return mlrun.common.schemas.IguazioProject(data=project.dict())

    def get_project_owner(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        project = self.get_project(session, name)
        return mlrun.common.schemas.ProjectOwner(
            username=project.spec.owner, access_key=self.project_owner_access_key
        )
