import os

from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import create_session, close_session
from mlrun.utils import logger
import alembic.config


def init_data() -> None:
    logger.info("Creating initial data")

    # run migrations on existing DB or create it with alembic
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    os.chdir(dir_path)
    alembic.config.main(
        argv=[
            # raise error to exit on a failure
            "--raiseerr",
            "upgrade",
            "head",
        ]
    )
    os.chdir(cwd)

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
