from fastapi import FastAPI

from mlrun.app.api.api import api_router
from mlrun.app.api.utils import submit
from mlrun.app.db.sqldb.session import SessionLocal
from mlrun.app.singletons import initialize_singletons, get_db
from mlrun.config import config
from mlrun.db import periodic
from mlrun.utils import logger

app = FastAPI(title="MLRun",
              description="Machine Learning automation and tracking",
              version=config.version,
              debug=config.httpdb.debug)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    logger.info("configuration dump\n%s", config.dump_yaml())

    initialize_singletons()

    task = periodic.Task()
    periodic.schedule(task, 60)

    _reschedule_tasks()


def _reschedule_tasks():
    db_session = None
    try:
        db_session = SessionLocal()
        for data in get_db().list_schedules(db_session):
            if "schedule" not in data:
                logger.warning("bad scheduler data - %s", data)
                continue
            submit(db_session, data)
    finally:
        db_session.close()
