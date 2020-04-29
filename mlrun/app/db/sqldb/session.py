from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mlrun.app.db.sqldb.models import Base
from mlrun.config import config

engine = create_engine(config.httpdb.dsn)
SessionLocal = sessionmaker(bind=engine)
