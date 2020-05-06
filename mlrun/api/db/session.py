from mlrun.api.db.sqldb.session import create_session as sqldb_create_session
from sqlalchemy.orm import Session
from mlrun.config import config


def create_session() -> Session:
    if config.httpdb.db_type == "sqldb":
        return sqldb_create_session()
    else:
        return None


def close_session(db_session):

    # will be None when it's filedb session
    if db_session is not None:
        db_session.close()
