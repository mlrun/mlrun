from typing import Generator

from mlrun.app.db.sqldb import SessionLocal


def get_db_session() -> Generator:
    try:
        db_session = SessionLocal()
        yield db_session
    finally:
        db_session.close()
