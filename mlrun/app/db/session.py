from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mlrun.app.db.sqldb import SQLDB
from mlrun.config import config

engine = create_engine(config.httpdb.dsn)
SessionLocal = sessionmaker(bind=engine)

dbInstance = None


def get_db_instance():
    global dbInstance
    if dbInstance is None:
        dbInstance = SQLDB('')
    return dbInstance
