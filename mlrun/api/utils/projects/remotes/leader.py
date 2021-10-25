import abc
import datetime
import typing

import mlrun.api.schemas


class Member(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self,
        session: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def update_project(
        self, session: str, name: str, project: mlrun.api.schemas.Project,
    ):
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def list_projects(
        self, session: str, updated_after: typing.Optional[datetime.datetime] = None,
    ) -> typing.Tuple[
        typing.List[mlrun.api.schemas.Project], typing.Optional[datetime.datetime]
    ]:
        pass

    @abc.abstractmethod
    def get_project(self, session: str, name: str,) -> mlrun.api.schemas.Project:
        pass

    @abc.abstractmethod
    def format_as_leader_project(
        self, project: mlrun.api.schemas.Project
    ) -> mlrun.api.schemas.IguazioProject:
        pass

    @abc.abstractmethod
    def get_project_owner(
        self, session: str, name: str,
    ) -> mlrun.api.schemas.ProjectOwner:
        pass
