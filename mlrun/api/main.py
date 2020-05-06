from fastapi import FastAPI

from mlrun.api.api.api import api_router
from mlrun.api.api.utils import submit
from mlrun.api.db.sqldb.session import create_session
from mlrun.api.singletons import initialize_singletons, get_db
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
        db_session = create_session()
        for data in get_db().list_schedules(db_session):
            if "schedule" not in data:
                logger.warning("bad scheduler data - %s", data)
                continue
            submit(db_session, data)
    finally:
        db_session.close()
