import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import mlrun.api.utils.singletons.db


class Consumer(mlrun.api.utils.projects.consumers.base.Consumer):
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.Project
    ):
        mlrun.api.utils.singletons.db.get_db().create_project(session, project)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        mlrun.api.utils.singletons.db.get_db().store_project(session, name, project)

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectPatch,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ):
        mlrun.api.utils.singletons.db.get_db().patch_project(
            session, name, project, patch_mode
        )

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        mlrun.api.utils.singletons.db.get_db().delete_project(session, name)

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return mlrun.api.utils.singletons.db.get_db().get_project(session, name)

    def list_projects(
        self, session: sqlalchemy.orm.Session, owner: str = None, full: bool = True,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return mlrun.api.utils.singletons.db.get_db().list_projects(
            session, owner, full
        )
