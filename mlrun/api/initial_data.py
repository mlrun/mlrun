import logging

from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import create_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    db_session = create_session()
    init_db(db_session)


def main() -> None:
    logger.info("Creating initial data")
    init()
    logger.info("Initial data created")


if __name__ == "__main__":
    main()
