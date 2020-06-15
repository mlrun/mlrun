import uvicorn
from fastapi import FastAPI

from mlrun.api.api.api import api_router
from mlrun.api.api.utils import submit
from mlrun.api.db.session import create_session, close_session
from mlrun.api.singletons import initialize_singletons, get_db
from mlrun.config import config
from mlrun.db import periodic
from mlrun.api.utils.periodic import run_function_periodically, cancel_periodic_functions
from mlrun.utils import logger
from mlrun.api.initial_data import init_data
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler

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

    _start_periodic_cleanup()


@app.on_event("shutdown")
async def shutdown_event():
    cancel_periodic_functions()


def _start_periodic_cleanup():
    logger.info('Starting periodic runtimes cleanup')
    run_function_periodically(int(config.runtimes_cleanup_interval), _cleanup_runtimes)


def _cleanup_runtimes():
    logger.debug('Cleaning runtimes')
    db_session = create_session()
    try:
        for kind in RuntimeKinds.runtime_with_handlers():
            runtime_handler = get_runtime_handler(kind)
            runtime_handler.delete_resources(get_db(), db_session)
    finally:
        close_session(db_session)


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
        close_session(db_session)


def main():
    init_data()
    uvicorn.run(
        'mlrun.api.main:app',
        host='0.0.0.0',
        port=config.httpdb.port,
        debug=config.httpdb.debug,
    )


if __name__ == '__main__':
    main()
