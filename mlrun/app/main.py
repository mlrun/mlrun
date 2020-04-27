from pathlib import Path

from fastapi import FastAPI

from mlrun.app.api.api import api_router
from mlrun.app.api.utils import submit
from mlrun.app.db.filedb.db import FileDB
from mlrun.app.db.sqldb.db import SQLDB
from mlrun.app.db.sqldb.session import SessionLocal
from mlrun.config import config
from mlrun.db import periodic
from mlrun.k8s_utils import K8sHelper
from mlrun.scheduler import Scheduler
from mlrun.utils import logger

app = FastAPI(title="MLRun",
              description="Machine Learning automation and tracking",
              version=config.version,
              debug=config.httpdb.debug)

app.include_router(api_router, prefix="/api")

scheduler: Scheduler = None
k8s: K8sHelper = None
logs_dir = None
db = None


@app.on_event("startup")
async def startup_event():
    global logs_dir, k8s, scheduler

    logger.info("configuration dump\n%s", config.dump_yaml())

    _initialize_db()

    logs_dir = Path(config.httpdb.logs_path)

    try:
        k8s = K8sHelper()
    except Exception:
        pass

    task = periodic.Task()
    periodic.schedule(task, 60)
    scheduler = Scheduler()

    _reschedule_tasks()


def _reschedule_tasks():
    global db
    try:
        db_session = SessionLocal()
        for data in db.list_schedules(db_session):
            if "schedule" not in data:
                logger.warning("bad scheduler data - %s", data)
                continue
            submit(db_session, data)
    finally:
        db_session.close()


def _initialize_db():
    global db

    if config.httpdb.db_type == "sqldb":
        logger.info("using SQLDB")
        db = SQLDB(config.httpdb.dsn)
        try:
            db_session = SessionLocal()
            db.initialize(db_session)
        finally:
            db_session.close()
    else:
        logger.info("using FileRunDB")
        db = FileDB(config.httpdb.dirpath)
        db.initialize(None)
