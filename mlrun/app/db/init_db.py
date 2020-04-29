from sqlalchemy.orm import Session

from mlrun.app.db.sqldb.models import Base
from mlrun.app.db.sqldb.session import engine


def init_db(db: Session) -> None:
    Base.metadata.create_all(bind=engine)