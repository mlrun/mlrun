import abc
import typing

import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.utils.singleton
from mlrun.utils import logger


class Member(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    def ensure_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        wait_for_completion: bool = True,
        leader_session: typing.Optional[str] = None,
    ):
        project_names = self.list_projects(
            db_session, format_=mlrun.api.schemas.Format.name_only
        )
        if name in project_names.projects:
            return
        logger.info(
            "Ensure project called, but project does not exist. Creating", name=name
        )
        project = mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=name),
        )
        self.create_project(
            db_session,
            project,
            leader_session=leader_session,
            wait_for_completion=wait_for_completion,
        )

    @abc.abstractmethod
    def create_project(
        self,
        db_session: sqlalchemy.orm.Session,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def store_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def patch_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> typing.Tuple[mlrun.api.schemas.Project, bool]:
        pass

    @abc.abstractmethod
    def delete_project(
        self,
        db_session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
        projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole] = None,
        leader_session: typing.Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_project(
        self, db_session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        pass

    @abc.abstractmethod
    def list_projects(
        self,
        db_session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: typing.List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        pass
