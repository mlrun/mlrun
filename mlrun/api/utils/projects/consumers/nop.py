import typing
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base


class Consumer(mlrun.api.utils.projects.consumers.base.Consumer):
    def __init__(self) -> None:
        super().__init__()
        self._projects: typing.Dict[str, mlrun.api.schemas.Project] = {}

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        self._projects[project.name] = mlrun.api.schemas.Project(**project.dict())

    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        self._projects[project.name].owner = project.owner

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        if name in self._projects:
            del self._projects[name]

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectOut:
        return mlrun.api.schemas.ProjectOut(project=self._projects[name])

    def list_projects(
        self, session: sqlalchemy.orm.Session
    ) -> mlrun.api.schemas.ProjectsOutput:
        return mlrun.api.schemas.ProjectsOutput(projects=self._projects.values())
