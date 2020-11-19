import typing

import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import mlrun.errors


class Consumer(mlrun.api.utils.projects.consumers.base.Consumer):
    def __init__(self) -> None:
        super().__init__()
        self._projects: typing.Dict[str, mlrun.api.schemas.Project] = {}

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        if project.name in self._projects:
            raise mlrun.errors.MLRunConflictError("Project already exists")
        self._projects[project.name] = mlrun.api.schemas.Project(**project.dict())

    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        self._projects[name] = self._projects[name].copy(
            update=project.dict(exclude_unset=True)
        )

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        if name in self._projects:
            del self._projects[name]

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return self._projects[name]

    def list_projects(
        self, session: sqlalchemy.orm.Session, owner: str = None, full: bool = True,
    ) -> mlrun.api.schemas.ProjectsOutput:
        if owner:
            raise NotImplementedError()
        if full:
            return mlrun.api.schemas.ProjectsOutput(
                projects=list(self._projects.values())
            )
        else:
            project_names = [project.name for project in list(self._projects.values())]
            return mlrun.api.schemas.ProjectsOutput(projects=project_names)
