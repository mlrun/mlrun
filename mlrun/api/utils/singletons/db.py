from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.session import create_session
from mlrun.config import config
from mlrun.utils import logger

# TODO: something nicer
db: DBInterface = None


def get_db() -> DBInterface:
    global db
    return db


def initialize_db(override_db=None):
    global db
    if override_db:
        db = override_db
        return
    logger.info("Creating sql db")
    db = SQLDB(config.httpdb.dsn)
    db_session = None
    try:
        db_session = create_session()
        db.initialize(db_session)
    finally:
        db_session.close()
