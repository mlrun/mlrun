import os
import pathlib

from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import create_session, close_session
from mlrun.utils import logger
from .utils.alembic import AlembicUtil


def init_data(from_scratch: bool = False) -> None:
    logger.info("Creating initial data")

    # run migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / "alembic.ini"

    alembic_util = AlembicUtil(alembic_config_path)
    alembic_util.init_alembic(from_scratch=from_scratch)

    db_session = create_session()
    try:
        init_db(db_session)
    finally:
        close_session(db_session)
    logger.info("Initial data created")


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
