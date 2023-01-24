import time
import traceback
import uuid

import fastapi
import uvicorn.protocols.utils
from starlette.middleware.base import BaseHTTPMiddleware

from mlrun.config import config
from mlrun.utils import logger


def init_middlewares(app: fastapi.FastAPI):
    for func in [
        log_request_response,
        ui_clear_cache,
        ensure_be_version,
    ]:
        app.add_middleware(BaseHTTPMiddleware, dispatch=func)


async def log_request_response(request: fastapi.Request, call_next):
    """
    This middleware logs request and response including its start / end time and duration
    """
    request_id = str(uuid.uuid4())
    silent_logging_paths = [
        "healthz",
    ]
    path_with_query_string = uvicorn.protocols.utils.get_path_with_query_string(
        request.scope
    )
    start_time = time.perf_counter_ns()
    if not any(
        silent_logging_path in path_with_query_string
        for silent_logging_path in silent_logging_paths
    ):
        logger.debug(
            "Received request",
            headers=request.headers,
            method=request.method,
            client_address=_resolve_client_address(request.scope),
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
            # it is starletter's ServerErrorMiddleware which is responsible for catching any un-handled exception
            # and transforming it to 500 response. therefore we can statically assign status code to 500
            status_code=500,
            request_id=request_id,
            uri=path_with_query_string,
            method=request.method,
            exc=exc,
            traceback=traceback.format_exc(),
        )
        raise
    else:
        # convert from nanoseconds to milliseconds
        elapsed_time_in_ms = (time.perf_counter_ns() - start_time) / 1000 / 1000
        if not any(
            silent_logging_path in path_with_query_string
            for silent_logging_path in silent_logging_paths
        ):
            logger.debug(
                "Sending response",
                status_code=response.status_code,
                request_id=request_id,
                elapsed_time_in_ms=elapsed_time_in_ms,
                uri=path_with_query_string,
                method=request.method,
                headers=response.headers,
            )
        return response


async def ui_clear_cache(request: fastapi.Request, call_next):
    """
    This middleware tells ui when to clear its cache based on backend version changes.
    """
    ui_version = request.headers.get("X-MLRun-UI-Version", "")
    response: fastapi.Response = await call_next(request)
    development_version = config.version.startswith("0.0.0")

    # do not ask ui to reload cache on development versions as it will make each request to reload ui and clear cache
    if not development_version and not ui_version or ui_version != config.version:

        # clear site cache
        response.headers["Clear-Site-Data"] = '"cache"'

        # tell ui to reload
        response.headers["X-MLRun-UI-Clear-Cache"] = "true"
    return response


async def ensure_be_version(request: fastapi.Request, call_next):
    """
    This middleware ensures response header includes backend version
    """
    response: fastapi.Response = await call_next(request)
    response.headers["X-MLRun-BE-Version"] = config.version
    return response


def _resolve_client_address(scope):
    # uvicorn expects this to be a tuple while starlette test client sets it to be a list
    if isinstance(scope.get("client"), list):
        scope["client"] = tuple(scope.get("client"))
    return uvicorn.protocols.utils.get_client_addr(scope)
