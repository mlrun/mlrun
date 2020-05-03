from sqlalchemy.orm import Session

from .sqldb.models import Base
from .sqldb.session import engine


def init_db(db: Session) -> None:
    Base.metadata.create_all(bind=engine)
