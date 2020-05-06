from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.models import Base
from mlrun.api.db.sqldb.session import get_engine


def init_db(db: Session) -> None:
    Base.metadata.create_all(bind=get_engine())
