import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base


class Consumer(mlrun.api.utils.projects.consumers.base.Consumer):
    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        pass

    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        pass

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        pass

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.ProjectOutput:
        pass

    def list_projects(
        self, session: sqlalchemy.orm.Session
    ) -> mlrun.api.schemas.ProjectsOutput:
        pass
