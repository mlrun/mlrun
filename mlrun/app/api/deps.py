from typing import Generator

from mlrun.app.db.sqldb.session import SessionLocal
from mlrun.config import config


def get_db_session() -> Generator:
    if config.httpdb.db_type == "sqldb":
        try:
            db_session = SessionLocal()
            yield db_session
        finally:
            db_session.close()
    else:
        yield None
