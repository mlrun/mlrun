import uuid

import fastapi
import fastapi.concurrency
import uvicorn
import uvicorn.protocols.utils
from fastapi.exception_handlers import http_exception_handler

import mlrun.errors
from mlrun.api.api.api import api_router
from mlrun.api.db.session import create_session, close_session
from mlrun.api.initial_data import init_data
from mlrun.api.utils.periodic import (
    run_function_periodically,
    cancel_periodic_functions,
)
from mlrun.api.utils.singletons.db import get_db, initialize_db
from mlrun.api.utils.singletons.logs_dir import initialize_logs_dir
from mlrun.api.utils.singletons.scheduler import initialize_scheduler, get_scheduler
from mlrun.config import config
from mlrun.k8s_utils import get_k8s_helper
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.utils import logger

app = fastapi.FastAPI(
    title="MLRun",
    description="Machine Learning automation and tracking",
    version=config.version,
    debug=config.httpdb.debug,
    # adding /api prefix
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    default_response_class=fastapi.responses.ORJSONResponse,
)

app.include_router(api_router, prefix="/api")


@app.exception_handler(mlrun.errors.MLRunHTTPStatusError)
async def http_status_error_handler(
    request: fastapi.Request, exc: mlrun.errors.MLRunHTTPStatusError
):
    status_code = exc.response.status_code
    error_message = repr(exc)
    logger.warning(
        "Request handling returned error status",
        error_message=error_message,
        status_code=status_code,
    )
    return await http_exception_handler(
        request, fastapi.HTTPException(status_code=status_code, detail=error_message)
    )


def get_client_address(scope):
    # uvicorn expects this to be a tuple while starlette test client sets it to be a list
    if isinstance(scope.get("client"), list):
        scope["client"] = tuple(scope.get("client"))
    return uvicorn.protocols.utils.get_client_addr(scope)


@app.middleware("http")
async def log_request_response(request: fastapi.Request, call_next):
    request_id = str(uuid.uuid4())
    silent_logging_paths = [
        "healthz",
    ]
    path_with_query_string = uvicorn.protocols.utils.get_path_with_query_string(
        request.scope
    )
    if not any(
        silent_logging_path in path_with_query_string
        for silent_logging_path in silent_logging_paths
    ):
        logger.debug(
            "Received request",
            method=request.method,
            client_address=get_client_address(request.scope),
            http_version=request.scope["http_version"],
            request_id=request_id,
            uri=path_with_query_string,
        )
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.warning(
            "Request handling failed. Sending response",
            # User middleware (like this one) runs after the exception handling middleware, the only thing running after
            # it is Starletter's ServerErrorMiddleware which is responsible for catching any un-handled exception
            # and transforming it to 500 response. therefore we can statically assign status code to 500
            status_code=500,
            request_id=request_id,
            uri=path_with_query_string,
            method=request.method,
            exc=exc,
        )
        raise
    else:
        if not any(
            silent_logging_path in path_with_query_string
            for silent_logging_path in silent_logging_paths
        ):
            logger.debug(
                "Sending response",
                status_code=response.status_code,
                request_id=request_id,
                uri=path_with_query_string,
                method=request.method,
            )
        return response


@app.on_event("startup")
async def startup_event():
    logger.info("configuration dump", dumped_config=config.dump_yaml())

    await _initialize_singletons()

    # periodic cleanup is not needed if we're not inside kubernetes cluster
    if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
        _start_periodic_cleanup()
        _start_periodic_runs_monitoring()


@app.on_event("shutdown")
async def shutdown_event():
    cancel_periodic_functions()
    await get_scheduler().stop()


async def _initialize_singletons():
    initialize_db()
    await initialize_scheduler()
    initialize_logs_dir()


def _start_periodic_cleanup():
    interval = int(config.runtimes_cleanup_interval)
    if interval > 0:
        logger.info("Starting periodic runtimes cleanup", interval=interval)
        run_function_periodically(interval, _cleanup_runtimes)


def _start_periodic_runs_monitoring():
    interval = int(config.runs_monitoring_interval)
    if interval > 0:
        logger.info("Starting periodic runs monitoring", interval=interval)
        run_function_periodically(interval, _monitor_runs)


def _monitor_runs():
    db_session = create_session()
    try:
        for kind in RuntimeKinds.runtime_with_handlers():
            try:
                runtime_handler = get_runtime_handler(kind)
                runtime_handler.monitor_runs(get_db(), db_session)
            except Exception as exc:
                logger.warning(
                    "Failed monitoring runs. Ignoring", exc=str(exc), kind=kind
                )
    finally:
        close_session(db_session)


def _cleanup_runtimes():
    logger.debug("Cleaning runtimes")
    db_session = create_session()
    try:
        for kind in RuntimeKinds.runtime_with_handlers():
            try:
                runtime_handler = get_runtime_handler(kind)
                runtime_handler.delete_resources(get_db(), db_session)
            except Exception as exc:
                logger.warning(
                    "Failed deleting resources. Ignoring", exc=str(exc), kind=kind
                )
    finally:
        close_session(db_session)


def main():
    init_data()
    uvicorn.run(
        "mlrun.api.main:app",
        host="0.0.0.0",
        port=config.httpdb.port,
        debug=config.httpdb.debug,
        access_log=False,
    )


if __name__ == "__main__":
    main()
