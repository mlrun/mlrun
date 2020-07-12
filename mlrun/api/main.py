import uvicorn
from fastapi import FastAPI

from mlrun.api.api.api import api_router
from mlrun.api.db.session import create_session, close_session
from mlrun.api.initial_data import init_data
from mlrun.api.utils.periodic import (
    run_function_periodically,
    cancel_periodic_functions,
)
from mlrun.api.utils.singletons.db import get_db, initialize_db
from mlrun.api.utils.singletons.k8s import initialize_k8s
from mlrun.api.utils.singletons.logs_dir import initialize_logs_dir
from mlrun.api.utils.singletons.scheduler import initialize_scheduler, get_scheduler
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.utils import logger

app = FastAPI(
    title="MLRun",
    description="Machine Learning automation and tracking",
    version=config.version,
    debug=config.httpdb.debug,
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    logger.info("configuration dump", dumped_config=config.dump_yaml())

    await _initialize_singletons()

    _start_periodic_cleanup()


@app.on_event("shutdown")
async def shutdown_event():
    cancel_periodic_functions()
    await get_scheduler().stop()


async def _initialize_singletons():
    initialize_db()
    await initialize_scheduler()
    initialize_k8s()
    initialize_logs_dir()


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
