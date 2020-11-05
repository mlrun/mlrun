import os
import pathlib

from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import create_session, close_session
from mlrun.utils import logger
import alembic.config


def init_data() -> None:
    logger.info("Creating initial data")

    # run migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / "alembic.ini"
    alembic.config.main(
        argv=[
            # raise error to exit on a failure
            "--raiseerr",
            "-c"
            f"{str(alembic_config_path)}",
            "upgrade",
            "head",
        ]
    )

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
