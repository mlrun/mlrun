from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.session import create_session as sqldb_create_session


def create_session() -> Session:
    return sqldb_create_session()


def close_session(db_session):
    db_session.close()


def run_function_with_new_db_session(func):
    session = create_session()
    try:
        result = func(session)
        return result
    finally:
        close_session(session)
