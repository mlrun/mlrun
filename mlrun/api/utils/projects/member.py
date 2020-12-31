import abc
import typing

import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.utils.singleton


class Member(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def ensure_project(self, session: sqlalchemy.orm.Session, name: str):
        pass

    @abc.abstractmethod
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.Project
    ) -> mlrun.api.schemas.Project:
        pass

    @abc.abstractmethod
    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        pass

    @abc.abstractmethod
    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ):
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy.default(),
    ):
        pass

    @abc.abstractmethod
    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        pass
