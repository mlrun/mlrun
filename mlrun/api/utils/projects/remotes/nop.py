import typing

import mergedeep
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.member
import mlrun.errors


class Member(mlrun.api.utils.projects.remotes.member.Member):
    def __init__(self) -> None:
        super().__init__()
        self._projects: typing.Dict[str, mlrun.api.schemas.Project] = {}

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.Project
    ):
        if project.name in self._projects:
            raise mlrun.errors.MLRunConflictError("Project already exists")
        self._projects[project.name] = project

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        self._projects[name] = project

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectPatch,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ):
        existing_project_dict = self._projects[name].dict()
        strategy = patch_mode.to_mergedeep_strategy()
        mergedeep.merge(
            existing_project_dict, project.dict(exclude_unset=True), strategy=strategy
        )
        self._projects[name] = mlrun.api.schemas.Project(**existing_project_dict)

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        if name in self._projects:
            del self._projects[name]

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return self._projects[name]

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
    ) -> mlrun.api.schemas.ProjectsOutput:
        if owner:
            raise NotImplementedError()
        if format_ == mlrun.api.schemas.Format.full:
            return mlrun.api.schemas.ProjectsOutput(
                projects=list(self._projects.values())
            )
        elif format_ == mlrun.api.schemas.Format.name_only:
            project_names = [project.name for project in list(self._projects.values())]
            return mlrun.api.schemas.ProjectsOutput(projects=project_names)
        else:
            raise NotImplementedError(
                f"Provided format is not supported. format={format_}"
            )
