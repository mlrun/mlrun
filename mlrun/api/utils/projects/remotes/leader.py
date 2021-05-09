import abc
import datetime
import typing

import mlrun.api.schemas


class Member(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self,
        session_cookie: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def store_project(
        self,
        session_cookie: str,
        name: str,
        project: mlrun.api.schemas.Project,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session_cookie: str,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def list_projects(
        self, session_cookie: str, updated_after: typing.Optional[datetime.datetime] = None
    ) -> typing.List[mlrun.api.schemas.Project]:
        pass
