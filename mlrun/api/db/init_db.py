from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.models import Base
from mlrun.api.db.sqldb.session import get_engine
from mlrun.config import config


def init_db(db_session: Session) -> None:
    if config.httpdb.db_type == "sqldb":
        Base.metadata.create_all(bind=get_engine())
