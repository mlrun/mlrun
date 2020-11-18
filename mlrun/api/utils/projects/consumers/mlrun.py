import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import mlrun.api.utils.singletons.db


class Consumer(mlrun.api.utils.projects.consumers.base.Consumer):
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        mlrun.api.utils.singletons.db.get_db().create_project(session, project)

    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        mlrun.api.utils.singletons.db.get_db().update_project(session, name, project)

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        mlrun.api.utils.singletons.db.get_db().delete_project(session, name)

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectOutput:
        return mlrun.api.utils.singletons.db.get_db().get_project(session, name)

    def list_projects(
        self, session: sqlalchemy.orm.Session, owner: str = None, full: bool = True,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return mlrun.api.utils.singletons.db.get_db().list_projects(
            session, owner, full
        )
