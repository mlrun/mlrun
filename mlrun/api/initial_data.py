import os
import sqlalchemy.orm
import pathlib

from mlrun.api.db.init_db import init_db
import mlrun.api.db.sqldb.db
import mlrun.api.schemas
from mlrun.api.db.session import create_session, close_session
from mlrun.utils import logger
from .utils.alembic import AlembicUtil


def init_data(from_scratch: bool = False) -> None:
    logger.info("Creating initial data")

    # run schema migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / "alembic.ini"

    alembic_util = AlembicUtil(alembic_config_path)
    alembic_util.init_alembic(from_scratch=from_scratch)

    db_session = create_session()
    try:
        init_db(db_session)
        _perform_data_migrations(db_session)
    finally:
        close_session(db_session)
    logger.info("Initial data created")


def _perform_data_migrations(db_session: sqlalchemy.orm.Session):
    # FileDB is not really a thing anymore, so using SQLDB directly
    db = mlrun.api.db.sqldb.db.SQLDB("")
    logger.info("Performing data migrations")
    _fill_project_state(db, db_session)


def _fill_project_state(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    projects = db.list_projects(db_session)
    for project in projects.projects:
        changed = False
        if not project.spec.desired_state:
            changed = True
            project.spec.desired_state = mlrun.api.schemas.ProjectState.online
        if not project.status.state:
            changed = True
            project.status.state = project.spec.desired_state
        if changed:
            logger.debug(
                "Found project without state data. Enriching",
                name=project.metadata.name,
            )
            db.store_project(db_session, project.metadata.name, project)


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
