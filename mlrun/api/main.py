import uuid

import fastapi
import uvicorn
import uvicorn.protocols.utils
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

import mlrun.errors
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
from mlrun.k8s_utils import get_k8s_helper
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
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


def get_client_addr(scope):
    # uvicorn expects this to be a tuple while starlette test client sets it to be a list
    if isinstance(scope.get("client"), list):
        scope["client"] = tuple(scope.get("client"))
    uvicorn.protocols.utils.get_client_addr(scope)


@app.middleware("http")
async def log_request_response(request: fastapi.Request, call_next):
    request_id = uuid.uuid4()
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
        logger.info(
            "Received request",
            method=request.method,
            client_address=get_client_addr(request.scope),
            http_version=request.scope["http_version"],
            request_id=request_id,
            uri=path_with_query_string,
        )
    try:
        response = await call_next(request)

    # Using StarletteHTTPException because
    # https://fastapi.tiangolo.com/tutorial/handling-errors/#fastapis-httpexception-vs-starlettes-httpexception
    except StarletteHTTPException as exc:
        logger.warning(
            "Request handling returned error status. Sending response",
            status_code=exc.status_code,
            request_id=request_id,
            uri=path_with_query_string,
            method=request.method,
        )
        raise
    except Exception:
        logger.warning(
            "Request handling failed. Sending response",
            # TODO: find a better solution
            # Basically any exception that not inherits from StarletteHTTPException will return a 500 status code
            # UNLESS a special exception handler was added for that exception, in this case this exception handler runs
            # after this middleware and if it will change the response code this 500 won't be true.
            # currently I can't find a way to execute code after all custom exception handlers
            status_code=500,
            request_id=request_id,
            uri=path_with_query_string,
            method=request.method,
        )
        raise
    else:
        if not any(
            silent_logging_path in path_with_query_string
            for silent_logging_path in silent_logging_paths
        ):
            logger.info(
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
    logger.info("Starting periodic runtimes cleanup")
    run_function_periodically(int(config.runtimes_cleanup_interval), _cleanup_runtimes)


def _cleanup_runtimes():
    logger.debug("Cleaning runtimes")
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
        "mlrun.api.main:app",
        host="0.0.0.0",
        port=config.httpdb.port,
        debug=config.httpdb.debug,
        access_log=False,
    )


if __name__ == "__main__":
    main()
