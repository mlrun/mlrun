import datetime
import typing

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.leader
import mlrun.api.utils.singletons.project_member
import mlrun.errors


class Member(mlrun.api.utils.projects.remotes.leader.Member):
    def __init__(self) -> None:
        super().__init__()
        self._project_role = mlrun.api.schemas.ProjectsRole.nop

    def create_project(
        self,
        session: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        return mlrun.api.utils.singletons.project_member.get_project_member().create_project(
            None, project, self._project_role
        )

    def store_project(
        self,
        session: str,
        name: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        return mlrun.api.utils.singletons.project_member.get_project_member().store_project(
            None, name, project, self._project_role
        )

    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        return mlrun.api.utils.singletons.project_member.get_project_member().delete_project(
            None, name, deletion_strategy, self._project_role
        )

    def list_projects(
        self, session: str, updated_after: typing.Optional[datetime.datetime] = None,
    ) -> typing.Tuple[
        typing.List[mlrun.api.schemas.Project], typing.Optional[datetime.datetime]
    ]:
        return (
            mlrun.api.utils.singletons.project_member.get_project_member()
            .list_projects(None)
            .projects,
            datetime.datetime.utcnow(),
        )

    def get_project(self, session: str, name: str,) -> mlrun.api.schemas.Project:
        return mlrun.api.utils.singletons.project_member.get_project_member().get_project(
            session, name
        )
