from sqlalchemy.orm import Session

from mlrun.api.utils.db.mysql import MySQLUtil
from mlrun.api.db.sqldb.models import Base as MySQLBase
from mlrun.api.db.sqldb.models_sqlite import Base as SQLiteBase
from mlrun.api.db.sqldb.session import get_engine
from mlrun.config import config


def init_db(db_session: Session) -> None:
    mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
    if mysql_dsn_data:
        base = MySQLBase
    else:
        base = SQLiteBase

    if config.httpdb.db_type != "filedb":
        base.metadata.create_all(bind=get_engine())
