import abc

import sqlalchemy.orm

import mlrun.api.schemas


class Consumer(abc.ABC):
    @abc.abstractmethod
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        pass

    @abc.abstractmethod
    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        pass

    @abc.abstractmethod
    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        pass

    @abc.abstractmethod
    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectOutput:
        pass

    @abc.abstractmethod
    def list_projects(
        self, session: sqlalchemy.orm.Session, owner: str = None, full: bool = True,
    ) -> mlrun.api.schemas.ProjectsOutput:
        pass
