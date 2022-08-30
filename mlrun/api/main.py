# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import asyncio
import concurrent.futures
import traceback
import uuid

import fastapi
import fastapi.concurrency
import uvicorn
import uvicorn.protocols.utils
from fastapi.exception_handlers import http_exception_handler

import mlrun.api.schemas
import mlrun.api.utils.clients.chief
import mlrun.errors
import mlrun.utils
import mlrun.utils.version
from mlrun.api.api.api import api_router
from mlrun.api.db.session import close_session, create_session
from mlrun.api.initial_data import init_data
from mlrun.api.utils.periodic import (
    cancel_all_periodic_functions,
    cancel_periodic_function,
    run_function_periodically,
)
from mlrun.api.utils.singletons.db import get_db, initialize_db
from mlrun.api.utils.singletons.logs_dir import initialize_logs_dir
from mlrun.api.utils.singletons.project_member import (
    get_project_member,
    initialize_project_member,
)
from mlrun.api.utils.singletons.scheduler import get_scheduler, initialize_scheduler
from mlrun.config import config
from mlrun.k8s_utils import get_k8s_helper
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.utils import logger

API_PREFIX = "/api"
BASE_VERSIONED_API_PREFIX = f"{API_PREFIX}/v1"


app = fastapi.FastAPI(
    title="MLRun",
    description="Machine Learning automation and tracking",
    version=config.version,
    debug=config.httpdb.debug,
    # adding /api prefix
    openapi_url=f"{BASE_VERSIONED_API_PREFIX}/openapi.json",
    docs_url=f"{BASE_VERSIONED_API_PREFIX}/docs",
    redoc_url=f"{BASE_VERSIONED_API_PREFIX}/redoc",
    default_response_class=fastapi.responses.ORJSONResponse,
)
app.include_router(api_router, prefix=BASE_VERSIONED_API_PREFIX)
# This is for backward compatibility, that is why we still leave it here but not include it in the schema
# so new users won't use the old un-versioned api
# TODO: remove when 0.9.x versions are no longer relevant
app.include_router(api_router, prefix=API_PREFIX, include_in_schema=False)


@app.exception_handler(Exception)
async def generic_error_handler(request: fastapi.Request, exc: Exception):
    error_message = repr(exc)
    return await fastapi.exception_handlers.http_exception_handler(
        # we have no specific knowledge on what was the exception and what status code fits so we simply use 500
        # This handler is mainly to put the error message in the right place in the body so the client will be able to
        # show it
        # TODO: 0.6.6 is the last version expecting the error details to be under reason, when it's no longer a relevant
        #  version can be changed to detail=error_message
        request,
        fastapi.HTTPException(status_code=500, detail={"reason": error_message}),
    )


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
        traceback=traceback.format_exc(),
    )
    # TODO: 0.6.6 is the last version expecting the error details to be under reason, when it's no longer a relevant
    #  version can be changed to detail=error_message
    return await http_exception_handler(
        request,
        fastapi.HTTPException(
            status_code=status_code, detail={"reason": error_message}
        ),
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
            traceback=traceback.format_exc(),
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
    logger.info(
        "configuration dump",
        dumped_config=config.dump_yaml(),
        version=mlrun.utils.version.Version().get(),
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=int(config.httpdb.max_workers)
        )
    )

    initialize_logs_dir()
    initialize_db()

    if (
        config.httpdb.clusterization.worker.sync_with_chief.mode
        == mlrun.api.schemas.WaitForChiefToReachOnlineStateFeatureFlag.enabled
        and config.httpdb.clusterization.role
        == mlrun.api.schemas.ClusterizationRole.worker
    ):
        _start_chief_clusterization_spec_sync_loop()

    if config.httpdb.state == mlrun.api.schemas.APIStates.online:
        await move_api_to_online()


@app.on_event("shutdown")
async def shutdown_event():
    if get_project_member():
        get_project_member().shutdown()
    cancel_all_periodic_functions()
    if get_scheduler():
        await get_scheduler().stop()


async def move_api_to_online():
    logger.info("Moving api to online")
    await initialize_scheduler()

    # In general it makes more sense to initialize the project member before the scheduler but in 1.1.0 in follower
    # we've added the full sync on the project member initialization (see code there for details) which might delete
    # projects which requires the scheduler to be set
    initialize_project_member()

    # maintenance periodic functions should only run on the chief instance
    if config.httpdb.clusterization.role == mlrun.api.schemas.ClusterizationRole.chief:
        # runs cleanup/monitoring is not needed if we're not inside kubernetes cluster
        if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
            _start_periodic_cleanup()
            _start_periodic_runs_monitoring()


def _start_periodic_cleanup():
    interval = int(config.runtimes_cleanup_interval)
    if interval > 0:
        logger.info("Starting periodic runtimes cleanup", interval=interval)
        run_function_periodically(
            interval, _cleanup_runtimes.__name__, False, _cleanup_runtimes
        )


def _start_periodic_runs_monitoring():
    interval = int(config.runs_monitoring_interval)
    if interval > 0:
        logger.info("Starting periodic runs monitoring", interval=interval)
        run_function_periodically(
            interval, _monitor_runs.__name__, False, _monitor_runs
        )


def _start_chief_clusterization_spec_sync_loop():
    interval = int(config.httpdb.clusterization.worker.sync_with_chief.interval)
    if interval > 0:
        logger.info("Starting chief clusterization spec sync loop", interval=interval)
        run_function_periodically(
            interval,
            _synchronize_with_chief_clusterization_spec.__name__,
            False,
            _synchronize_with_chief_clusterization_spec,
        )


async def _synchronize_with_chief_clusterization_spec():
    # sanity
    # if we are still in the periodic function and the worker has reached the terminal state, then cancel it
    if config.httpdb.state in mlrun.api.schemas.APIStates.terminal_states():
        cancel_periodic_function(_synchronize_with_chief_clusterization_spec.__name__)

    try:
        chief_client = mlrun.api.utils.clients.chief.Client()
        clusterization_spec = chief_client.get_clusterization_spec(
            return_fastapi_response=False, raise_on_failure=True
        )
    except Exception as exc:
        logger.debug(
            "Failed receiving clusterization spec",
            exc=str(exc),
            traceback=traceback.format_exc(),
        )
    else:
        await _align_worker_state_with_chief_state(clusterization_spec)


async def _align_worker_state_with_chief_state(
    clusterization_spec: mlrun.api.schemas.ClusterizationSpec,
):
    chief_state = clusterization_spec.chief_api_state
    if not chief_state:
        logger.warning("Chief did not return any state")
        return

    if chief_state not in mlrun.api.schemas.APIStates.terminal_states():
        logger.debug(
            "Chief did not reach online state yet, will retry after sync interval",
            interval=config.httpdb.clusterization.worker.sync_with_chief.interval,
            chief_state=chief_state,
        )
        # we want the worker to be aligned with chief state
        config.httpdb.state = chief_state
        return

    if chief_state == mlrun.api.schemas.APIStates.online:
        logger.info("Chief reached online state! Switching worker state to online")
        await move_api_to_online()
        logger.info("Worker state reached online")

    else:
        logger.info(
            "Chief state is terminal, canceling worker periodic chief clusterization spec pulling",
            state=config.httpdb.state,
        )
    config.httpdb.state = chief_state
    # if reached terminal state we cancel the periodic function
    # assumption: we can't get out of a terminal api state, so no need to continue pulling when reached one
    cancel_periodic_function(_synchronize_with_chief_clusterization_spec.__name__)


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
    if config.httpdb.clusterization.role == mlrun.api.schemas.ClusterizationRole.chief:
        init_data()
    elif (
        config.httpdb.clusterization.worker.sync_with_chief.mode
        == mlrun.api.schemas.WaitForChiefToReachOnlineStateFeatureFlag.enabled
        and config.httpdb.clusterization.role
        == mlrun.api.schemas.ClusterizationRole.worker
    ):
        # we set this state to mark the phase between the startup of the instance until we able to pull the chief state
        config.httpdb.state = mlrun.api.schemas.APIStates.waiting_for_chief

    logger.info(
        "Starting API server",
        port=config.httpdb.port,
        debug=config.httpdb.debug,
    )
    uvicorn.run(
        "mlrun.api.main:app",
        host="0.0.0.0",
        port=config.httpdb.port,
        debug=config.httpdb.debug,
        access_log=False,
    )


if __name__ == "__main__":
    main()
