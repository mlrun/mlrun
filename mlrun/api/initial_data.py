import logging

from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import create_session, close_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_data() -> None:
    logger.info("Creating initial data")
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
