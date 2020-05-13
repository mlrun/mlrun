from pathlib import Path

from mlrun.api.db.base import DBInterface
from mlrun.api.db.filedb.db import FileDB
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.session import create_session
from mlrun.config import config
from mlrun.k8s_utils import K8sHelper
from mlrun.scheduler import Scheduler
from mlrun.utils import logger

# TODO: something nicer
scheduler: Scheduler = None
k8s: K8sHelper = None
logs_dir: Path = None
db: DBInterface = None


def initialize_singletons():
    _initialize_scheduler()
    _initialize_k8s()
    _initialize_logs_dir()
    _initialize_db()


def _initialize_scheduler():
    global scheduler
    scheduler = Scheduler()


def _initialize_k8s():
    global k8s
    try:
        k8s = K8sHelper()
    except Exception:
        pass


def _initialize_logs_dir():
    global logs_dir
    logs_dir = Path(config.httpdb.logs_path)


def _initialize_db():
    global db
    if config.httpdb.db_type == "sqldb":
        logger.info("using SQLDB")
        db = SQLDB(config.httpdb.dsn)
        db_session = None
        try:
            db_session = create_session()
            db.initialize(db_session)
        finally:
            db_session.close()
    else:
        logger.info("using FileRunDB")
        db = FileDB(config.httpdb.dirpath)
        db.initialize(None)


def get_db() -> DBInterface:
    global db
    return db


def get_logs_dir() -> Path:
    global logs_dir
    return logs_dir


def get_k8s() -> K8sHelper:
    global k8s
    return k8s


def get_scheduler() -> Scheduler:
    global scheduler
    return scheduler
