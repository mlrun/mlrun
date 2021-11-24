from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.session import create_session as sqldb_create_session
from mlrun.config import config


def create_session(db_type=None) -> Session:
    db_type = db_type or config.httpdb.db_type
    if db_type == "filedb":
        return None
    else:
        return sqldb_create_session()


def close_session(db_session):

    # will be None when it's filedb session
    if db_session is not None:
        db_session.close()


def run_function_with_new_db_session(func):
    session = create_session()
    try:
        result = func(session)
        return result
    finally:
        close_session(session)
