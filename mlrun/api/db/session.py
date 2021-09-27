from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.session import create_session as sqldb_create_session


def create_session() -> Session:
    return sqldb_create_session()


def close_session(db_session):
    db_session.close()
