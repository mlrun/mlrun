# Copyright 2023 Iguazio
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
import collections
import concurrent.futures
import contextlib
import datetime
import traceback
import typing

import fastapi
import fastapi.concurrency
import sqlalchemy.orm
from fastapi.exception_handlers import http_exception_handler

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.errors
import mlrun.lists
import mlrun.utils
import mlrun.utils.notifications
import mlrun.utils.version
import server.api.api.utils
import server.api.constants
import server.api.crud
import server.api.db.base
import server.api.initial_data
import server.api.middlewares
import server.api.runtime_handlers
import server.api.utils.clients.chief
import server.api.utils.clients.log_collector
import server.api.utils.notification_pusher
import server.api.utils.time_window_tracker
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.runtimes import RuntimeClassMode, RuntimeKinds
from mlrun.utils import logger
from server.api.api.api import api_router, api_v2_router
from server.api.db.session import close_session, create_session
from server.api.runtime_handlers import get_runtime_handler
from server.api.utils.periodic import (
    cancel_all_periodic_functions,
    cancel_periodic_function,
    run_function_periodically,
)
from server.api.utils.singletons.db import get_db, initialize_db
from server.api.utils.singletons.k8s import get_k8s_helper
from server.api.utils.singletons.logs_dir import initialize_logs_dir
from server.api.utils.singletons.project_member import (
    get_project_member,
    initialize_project_member,
)
from server.api.utils.singletons.scheduler import (
    ensure_scheduler,
    get_scheduler,
    start_scheduler,
)

API_PREFIX = "/api"
BASE_VERSIONED_API_PREFIX = f"{API_PREFIX}/v1"
V2_API_PREFIX = f"{API_PREFIX}/v2"

# This is a dictionary which holds the number of consecutive start log requests for each run uid.
# We use this dictionary to make sure that we don't get stuck in an endless loop of trying to collect logs for a runs
# that keep failing start logs requests.
_run_uid_start_log_request_counters: collections.Counter = collections.Counter()


# https://fastapi.tiangolo.com/advanced/events/
@contextlib.asynccontextmanager
async def lifespan(app_: fastapi.FastAPI):
    await setup_api()

    # Let the api run
    yield

    await teardown_api()


app = fastapi.FastAPI(
    title="MLRun",
    description="Machine Learning automation and tracking",
    version=config.version,
    debug=config.httpdb.debug,
    # adding /api prefix
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    default_response_class=fastapi.responses.ORJSONResponse,
    lifespan=lifespan,
)
app.include_router(api_router, prefix=BASE_VERSIONED_API_PREFIX)
app.include_router(api_v2_router, prefix=V2_API_PREFIX)
# This is for backward compatibility, that is why we still leave it here but not include it in the schema
# so new users won't use the old un-versioned api.
# /api points to /api/v1 since it is used externally, and we don't want to break it.
# TODO: make sure UI and all relevant Iguazio versions uses /api/v1 and deprecate this
app.include_router(api_router, prefix=API_PREFIX, include_in_schema=False)

# middlewares, order matter
app.add_middleware(
    server.api.middlewares.EnsureBackendVersionMiddleware,
    backend_version=config.version,
)
app.add_middleware(
    server.api.middlewares.UiClearCacheMiddleware, backend_version=config.version
)
app.add_middleware(server.api.middlewares.RequestLoggerMiddleware, logger=logger)


@app.exception_handler(Exception)
async def generic_error_handler(request: fastapi.Request, exc: Exception):
    error_message = repr(exc)
    return await fastapi.exception_handlers.http_exception_handler(
        # we have no specific knowledge on what was the exception and what status code fits so we simply use 500
        # This handler is mainly to put the error message in the right place in the body so the client will be able to
        # show it
        request,
        fastapi.HTTPException(status_code=500, detail=error_message),
    )


@app.exception_handler(mlrun.errors.MLRunHTTPStatusError)
async def http_status_error_handler(
    request: fastapi.Request, exc: mlrun.errors.MLRunHTTPStatusError
):
    request_id = None

    # request might not have request id when the error is raised before the request id is set on middleware
    if hasattr(request.state, "request_id"):
        request_id = request.state.request_id
    status_code = exc.response.status_code
    error_message = repr(exc)
    log_message = "Request handling returned error status"

    if isinstance(exc, mlrun.errors.EXPECTED_ERRORS):
        logger.debug(
            log_message,
            error_message=error_message,
            status_code=status_code,
            request_id=request_id,
        )
    else:
        logger.warning(
            log_message,
            error_message=error_message,
            status_code=status_code,
            traceback=traceback.format_exc(),
            request_id=request_id,
        )

    return await http_exception_handler(
        request,
        fastapi.HTTPException(status_code=status_code, detail=error_message),
    )


async def setup_api():
    logger.info(
        "On startup event handler called",
        config=config.dump_yaml(),
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

    # chief do stuff
    if (
        config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.chief
    ):
        server.api.initial_data.init_data()

    # worker
    elif (
        config.httpdb.clusterization.worker.sync_with_chief.mode
        == mlrun.common.schemas.WaitForChiefToReachOnlineStateFeatureFlag.enabled
        and config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.worker
    ):
        # in the background, wait for chief to reach online state
        _start_chief_clusterization_spec_sync_loop()

    if config.httpdb.state == mlrun.common.schemas.APIStates.online:
        await move_api_to_online()


async def teardown_api():
    if get_project_member():
        get_project_member().shutdown()
    cancel_all_periodic_functions()
    if get_scheduler():
        await get_scheduler().stop()


async def move_api_to_online():
    logger.info("Moving api to online")

    # scheduler is needed on both workers and chief
    # on workers - it allows to us to list/get scheduler(s)
    # on chief - it allows to us to create/update/delete schedule(s)
    ensure_scheduler()
    if (
        config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.chief
        and config.httpdb.clusterization.chief.feature_gates.scheduler == "enabled"
    ):
        await start_scheduler()

    # In general, it makes more sense to initialize the project member before the scheduler but in 1.1.0 in follower
    # we've added the full sync on the project member initialization (see code there for details) which might delete
    # projects which requires the scheduler to be set
    await fastapi.concurrency.run_in_threadpool(initialize_project_member)
    get_project_member().start()

    # maintenance periodic functions should only run on the chief instance
    if (
        config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.chief
    ):
        server.api.initial_data.update_default_configuration_data()
        # runs cleanup/monitoring is not needed if we're not inside kubernetes cluster
        if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
            if config.httpdb.clusterization.chief.feature_gates.cleanup == "enabled":
                _start_periodic_cleanup()
            if (
                config.httpdb.clusterization.chief.feature_gates.runs_monitoring
                == "enabled"
            ):
                _start_periodic_runs_monitoring()
            if (
                config.httpdb.clusterization.chief.feature_gates.pagination_cache
                == "enabled"
            ):
                _start_periodic_pagination_cache_monitoring()
            if (
                config.httpdb.clusterization.chief.feature_gates.project_summaries
                == "enabled"
            ):
                _start_periodic_project_summaries_calculation()
            if config.httpdb.clusterization.chief.feature_gates.start_logs == "enabled":
                await _start_periodic_logs_collection()
            if config.httpdb.clusterization.chief.feature_gates.stop_logs == "enabled":
                await _start_periodic_stop_logs()


async def _start_periodic_logs_collection():
    if config.log_collector.mode == mlrun.common.schemas.LogsCollectorMode.legacy:
        logger.info(
            "Using legacy logs collection method, skipping logs collection periodic function",
            mode=config.log_collector.mode,
        )
        return
    logger.info(
        "Starting logs collection periodic function",
        mode=config.log_collector.mode,
        interval=config.log_collector.periodic_start_log_interval,
    )
    start_logs_limit = asyncio.Semaphore(
        config.log_collector.concurrent_start_logs_workers
    )

    await _verify_log_collection_started_on_startup(start_logs_limit)

    run_function_periodically(
        interval=int(config.log_collector.periodic_start_log_interval),
        name=_initiate_logs_collection.__name__,
        replace=False,
        function=_initiate_logs_collection,
        start_logs_limit=start_logs_limit,
    )


async def _verify_log_collection_started_on_startup(
    start_logs_limit: asyncio.Semaphore,
):
    """
    Verifies that log collection was started on startup for all runs which might have started before the API
    initialization or after upgrade.
    In that case we want to make sure that all runs which are in non-terminal state will have their logs collected
    by the log-collector and runs which might have reached terminal state while the API was down will have their
    logs collected as well.
    If the amount of runs which require logs collection on startup exceeds the configured limit, we will skip the rest
    but mark them as requested logs collection, to not get the API stuck in an endless loop of trying to collect logs.
    :param start_logs_limit: Semaphore which limits the number of concurrent log collection tasks
    """
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    log_collection_cycle_tracker = server.api.utils.time_window_tracker.TimeWindowTracker(
        key=server.api.utils.time_window_tracker.TimeWindowTrackerKeys.log_collection,
        # If the API was down for more than the grace period, we will only collect logs for runs which reached
        # terminal state within the grace period and not since the API actually went down.
        max_window_size_seconds=min(
            int(config.log_collector.api_downtime_grace_period),
            int(config.runtime_resources_deletion_grace_period),
        ),
    )
    await fastapi.concurrency.run_in_threadpool(
        log_collection_cycle_tracker.initialize, db_session
    )
    last_update_time = await fastapi.concurrency.run_in_threadpool(
        log_collection_cycle_tracker.get_window, db_session
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        logger.debug(
            "Getting all runs which are in non terminal state and require logs collection"
        )
        runs_uids = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[None, False],
            only_uids=True,
            states=mlrun.common.runtimes.constants.RunStates.non_terminal_states(),
        )
        logger.debug(
            "Getting all runs which might have reached terminal state while the API was down",
            api_downtime_grace_period=config.log_collector.api_downtime_grace_period,
        )
        runs_uids.extend(
            await fastapi.concurrency.run_in_threadpool(
                get_db().list_distinct_runs_uids,
                db_session,
                requested_logs_modes=[None, False],
                # get only uids as there might be many runs which reached terminal state while the API was down, the
                # run objects will be fetched in the next step
                only_uids=True,
                last_update_time_from=last_update_time,
                states=mlrun.common.runtimes.constants.RunStates.terminal_states(),
            )
        )
        if runs_uids:
            skipped_run_uids = []
            if len(runs_uids) > int(
                mlrun.mlconf.log_collector.start_logs_startup_run_limit
            ):
                logger.warning(
                    "Amount of runs requiring logs collection on startup exceeds configured limit, "
                    "skipping the rest but marking them as requested",
                    total_runs_count=len(runs_uids),
                    start_logs_startup_run_limit=mlrun.mlconf.log_collector.start_logs_startup_run_limit,
                )
                skipped_run_uids = runs_uids[
                    int(mlrun.mlconf.log_collector.start_logs_startup_run_limit) :
                ]
                runs_uids = runs_uids[
                    : int(mlrun.mlconf.log_collector.start_logs_startup_run_limit)
                ]

            logger.debug(
                "Found runs which require logs collection on startup",
                runs_count=len(runs_uids),
            )

            # we're using best_effort=True so the api will mark the runs as requested logs collection even in cases
            # where the log collection failed (e.g. when the pod is not found for runs that might have reached terminal
            # state while the API was down)
            await _start_log_and_update_runs(
                start_logs_limit=start_logs_limit,
                db_session=db_session,
                runs_uids=runs_uids,
                best_effort=True,
            )

            if skipped_run_uids:
                await fastapi.concurrency.run_in_threadpool(
                    get_db().update_runs_requested_logs,
                    db_session,
                    uids=skipped_run_uids,
                    requested_logs=True,
                )
    finally:
        await fastapi.concurrency.run_in_threadpool(
            log_collection_cycle_tracker.update_window, db_session, now
        )
        await fastapi.concurrency.run_in_threadpool(close_session, db_session)


async def _initiate_logs_collection(start_logs_limit: asyncio.Semaphore):
    """
    This function is responsible for initiating the logs collection process. It will get a list of all runs which are
    in a state which requires logs collection and will initiate the logs collection process for each of them.
    :param start_logs_limit: a semaphore which limits the number of concurrent logs collection processes
    """
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    try:
        # list all the runs currently still running in the system which we didn't request logs collection for yet
        runs_uids = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[False],
            only_uids=True,
            states=mlrun.common.runtimes.constants.RunStates.non_terminal_states(),
        )

        last_update_time = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(
            seconds=int(config.runtime_resources_deletion_grace_period)
        )

        # Add all the completed/failed runs in the system which we didn't request logs collection for yet.
        # Aborted means the pods were deleted and logs were already fetched.
        run_states = mlrun.common.runtimes.constants.RunStates.terminal_states()
        run_states.remove(mlrun.common.runtimes.constants.RunStates.aborted)
        runs_uids.extend(
            await fastapi.concurrency.run_in_threadpool(
                get_db().list_distinct_runs_uids,
                db_session,
                requested_logs_modes=[False],
                only_uids=True,
                last_update_time_from=last_update_time,
                states=run_states,
            )
        )
        if runs_uids:
            logger.debug(
                "Found runs which require logs collection",
                runs_uids=len(runs_uids),
            )
            await _start_log_and_update_runs(
                start_logs_limit=start_logs_limit,
                db_session=db_session,
                runs_uids=runs_uids,
            )

    finally:
        await fastapi.concurrency.run_in_threadpool(close_session, db_session)


async def _start_log_and_update_runs(
    start_logs_limit: asyncio.Semaphore,
    db_session: sqlalchemy.orm.Session,
    runs_uids: list[str],
    best_effort: bool = False,
):
    if not runs_uids:
        return

    # get the runs from the DB
    runs = await fastapi.concurrency.run_in_threadpool(
        get_db().list_runs,
        db_session,
        uid=runs_uids,
        project="*",
    )

    # the max number of consecutive start log requests for a run before we mark it as requested logs collection
    # basically represents the grace period before the run's resources are deleted
    max_consecutive_start_log_requests = int(
        int(config.log_collector.failed_runs_grace_period)
        / int(config.log_collector.periodic_start_log_interval)
    )

    global _run_uid_start_log_request_counters
    runs_to_mark_as_requested_logs = []
    start_logs_for_runs = []
    for run in runs:
        run_uid = run.get("metadata", {}).get("uid", None)

        # if we requested logs for the same run more times than the threshold, we mark it as requested logs collection,
        # so the API and the log collector won't be stuck in an endless loop of trying to collect logs for it
        if (
            run_uid in _run_uid_start_log_request_counters
            and _run_uid_start_log_request_counters[run_uid]
            >= max_consecutive_start_log_requests
        ):
            logger.warning(
                "Run reached max consecutive start log requests, marking it as requested logs collection",
                run_uid=run_uid,
                requests_count=_run_uid_start_log_request_counters[run_uid],
            )
            runs_to_mark_as_requested_logs.append(run_uid)
            continue

        start_logs_for_runs.append(
            _start_log_for_run(
                run, start_logs_limit, raise_on_error=False, best_effort=best_effort
            )
        )
        if run_uid:
            _run_uid_start_log_request_counters.setdefault(run_uid, 0)
            _run_uid_start_log_request_counters[run_uid] += 1

    # each result contains either run_uid or None
    # if it's None it means something went wrong, and we should skip it
    # if it's run_uid it means we requested logs collection for it and we should update it's requested_logs field
    results = await asyncio.gather(*start_logs_for_runs, return_exceptions=True)
    successful_run_uids = [result for result in results if result]

    # distinct the runs uids
    runs_to_mark_as_requested_logs = list(
        set(runs_to_mark_as_requested_logs + successful_run_uids)
    )

    if len(runs_to_mark_as_requested_logs) > 0:
        logger.debug(
            "Updating runs to indicate that we requested logs collection for them",
            runs_uids=runs_to_mark_as_requested_logs,
        )
        # update the runs to indicate that we have requested log collection for them
        await fastapi.concurrency.run_in_threadpool(
            get_db().update_runs_requested_logs,
            db_session,
            uids=runs_to_mark_as_requested_logs,
        )

        # remove the counters for the runs we updated
        for run_uid in runs_to_mark_as_requested_logs:
            _run_uid_start_log_request_counters.pop(run_uid, None)


async def _start_log_for_run(
    run: dict,
    start_logs_limit: asyncio.Semaphore = None,
    raise_on_error: bool = True,
    best_effort: bool = False,
) -> typing.Optional[typing.Union[str, None]]:
    """
    Starts log collection for a specific run
    :param run: run object
    :param start_logs_limit: semaphore to limit the number of concurrent log collection requests
    :param raise_on_error: if True, will raise an exception if something went wrong, otherwise will return None and
    log the error
    :return: the run_uid of the run if log collection was started, None otherwise
    """
    # using semaphore to limit the number of concurrent log collection requests
    # this is to prevent opening too many connections to many connections
    async with start_logs_limit:
        logs_collector_client = (
            server.api.utils.clients.log_collector.LogCollectorClient()
        )
        run_kind = run.get("metadata", {}).get("labels", {}).get("kind", None)
        project_name = run.get("metadata", {}).get("project", None)
        run_uid = run.get("metadata", {}).get("uid", None)

        # information for why runtime isn't log collectable is inside the method
        if not mlrun.runtimes.RuntimeKinds.is_log_collectable_runtime(run_kind):
            # we mark the run as requested logs collection so we won't iterate over it again
            return run_uid
        try:
            runtime_handler: server.api.runtime_handlers.BaseRuntimeHandler = (
                await fastapi.concurrency.run_in_threadpool(
                    get_runtime_handler, run_kind
                )
            )
            object_id = runtime_handler.resolve_object_id(run)
            label_selector = runtime_handler.resolve_label_selector(
                project=project_name,
                object_id=object_id,
                class_mode=RuntimeClassMode.run,
                # when collecting logs for runtimes we only collect for the main runtime resource, as there could be
                # runtimes that the user will create with hundreds of resources (e.g mpi job can have multiple workers
                # which aren't really important for log collection
                with_main_runtime_resource_label_selector=True,
            )
            success, _ = await logs_collector_client.start_logs(
                run_uid=run_uid,
                selector=label_selector,
                project=project_name,
                best_effort=best_effort,
                raise_on_error=True,
            )
            if success:
                # update the run to mark that we requested logs collection for it
                return run_uid

        except Exception as exc:
            if raise_on_error:
                raise exc

            logger.warning(
                "Failed to start logs for run",
                run_uid=run_uid,
                exc=mlrun.errors.err_to_str(exc),
            )
            return None


def _start_periodic_cleanup():
    interval = int(config.runtimes_cleanup_interval)
    if interval > 0:
        logger.info("Starting periodic runtimes cleanup", interval=interval)
        run_function_periodically(
            interval, _cleanup_runtimes.__name__, False, _cleanup_runtimes
        )


def _start_periodic_runs_monitoring():
    interval = int(config.monitoring.runs.interval)
    if interval > 0:
        logger.info("Starting periodic runs monitoring", interval=interval)
        run_function_periodically(
            interval, _monitor_runs.__name__, False, _monitor_runs
        )


def _start_periodic_pagination_cache_monitoring():
    interval = int(config.httpdb.pagination.pagination_cache.interval)
    if interval > 0:
        logger.info("Starting periodic pagination cache monitoring", interval=interval)
        run_function_periodically(
            interval,
            server.api.crud.pagination_cache.PaginationCache().monitor_pagination_cache.__name__,
            False,
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.pagination_cache.PaginationCache().monitor_pagination_cache,
        )


def _start_periodic_project_summaries_calculation():
    interval = int(config.monitoring.projects.summaries.cache_interval)
    if interval > 0:
        logger.info(
            "Starting periodic project summaries calculation", interval=interval
        )
        run_function_periodically(
            interval,
            server.api.crud.projects.Projects().refresh_project_resources_counters_cache.__name__,
            False,
            server.api.db.session.run_async_function_with_new_db_session,
            server.api.crud.projects.Projects().refresh_project_resources_counters_cache,
        )


async def _start_periodic_stop_logs():
    if config.log_collector.mode == mlrun.common.schemas.LogsCollectorMode.legacy:
        logger.info(
            "Using legacy logs collection method, skipping stop logs periodic function",
            mode=config.log_collector.mode,
        )
        return

    await _verify_log_collection_stopped_on_startup()

    interval = int(config.log_collector.stop_logs_interval)
    if interval > 0:
        logger.info("Starting periodic stop logs", interval=interval)
        run_function_periodically(interval, _stop_logs.__name__, False, _stop_logs)


async def _verify_log_collection_stopped_on_startup():
    """
    First, list runs that are currently being collected in the log collector.
    Second, query the DB for those runs that are also in terminal state and have logs requested.
    Lastly, call stop logs for the runs that met all of the above conditions.
    This is done so that the log collector won't keep trying to collect logs for runs that are already
    in terminal state.
    """
    logger.debug("Listing runs currently being log collected")
    log_collector_client = server.api.utils.clients.log_collector.LogCollectorClient()
    run_uids_in_progress = []
    failed_listing = False
    try:
        runs_in_progress_response_stream = log_collector_client.list_runs_in_progress()
        # collate the run uids from the response stream to a list
        async for run_uids in runs_in_progress_response_stream:
            run_uids_in_progress.extend(run_uids)
    except Exception as exc:
        failed_listing = True
        logger.warning(
            "Failed listing runs currently being log collected",
            exc=err_to_str(exc),
            traceback=traceback.format_exc(),
        )

    if len(run_uids_in_progress) == 0 and not failed_listing:
        logger.debug("No runs currently being log collected")
        return

    logger.debug(
        "Getting current log collected runs which have reached terminal state and already have logs requested",
        run_uids_in_progress_count=len(run_uids_in_progress),
    )
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    try:
        runs = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[True],
            only_uids=False,
            states=mlrun.common.runtimes.constants.RunStates.terminal_states()
            + [
                # add unknown state as well, as it's possible that the run reached such state
                # usually it happens when run pods get preempted
                mlrun.common.runtimes.constants.RunStates.unknown,
            ],
            specific_uids=run_uids_in_progress,
        )

        if len(runs) > 0:
            logger.debug(
                "Stopping logs for runs which reached terminal state before startup",
                runs_count=len(runs),
            )
            await _stop_logs_for_runs(runs)
    finally:
        await fastapi.concurrency.run_in_threadpool(close_session, db_session)


def _start_chief_clusterization_spec_sync_loop():
    # put it here first, because we need to set it before the periodic function starts
    # so the worker will be aligned with the chief state
    config.httpdb.state = mlrun.common.schemas.APIStates.waiting_for_chief

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
    if config.httpdb.state in mlrun.common.schemas.APIStates.terminal_states():
        cancel_periodic_function(_synchronize_with_chief_clusterization_spec.__name__)

    try:
        chief_client = server.api.utils.clients.chief.Client()
        clusterization_spec = await chief_client.get_clusterization_spec(
            return_fastapi_response=False, raise_on_failure=True
        )
    except Exception as exc:
        logger.debug(
            "Failed receiving clusterization spec",
            exc=err_to_str(exc),
            traceback=traceback.format_exc(),
        )
    else:
        await _align_worker_state_with_chief_state(clusterization_spec)


async def _align_worker_state_with_chief_state(
    clusterization_spec: mlrun.common.schemas.ClusterizationSpec,
):
    chief_state = clusterization_spec.chief_api_state
    if not chief_state:
        logger.warning("Chief did not return any state")
        return

    if chief_state not in mlrun.common.schemas.APIStates.terminal_states():
        logger.debug(
            "Chief did not reach online state yet, will retry after sync interval",
            interval=config.httpdb.clusterization.worker.sync_with_chief.interval,
            chief_state=chief_state,
        )
        # we want the worker to be aligned with chief state
        config.httpdb.state = chief_state
        return

    if chief_state == mlrun.common.schemas.APIStates.online:
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


async def _monitor_runs():
    stale_runs = await fastapi.concurrency.run_in_threadpool(
        server.api.db.session.run_function_with_new_db_session,
        _monitor_runs_and_push_terminal_notifications,
    )
    await _abort_stale_runs(stale_runs)


def _monitor_runs_and_push_terminal_notifications(db_session):
    db = get_db()
    stale_runs = []
    for kind in RuntimeKinds.runtime_with_handlers():
        try:
            runtime_handler = get_runtime_handler(kind)
            runtime_stale_runs = runtime_handler.monitor_runs(db, db_session)
            stale_runs.extend(runtime_stale_runs)
        except Exception as exc:
            logger.warning(
                "Failed monitoring runs. Ignoring",
                exc=err_to_str(exc),
                kind=kind,
            )

    try:
        runs_monitoring_cycle_tracker = server.api.utils.time_window_tracker.TimeWindowTracker(
            key=server.api.utils.time_window_tracker.TimeWindowTrackerKeys.run_monitoring,
            max_window_size_seconds=int(config.runtime_resources_deletion_grace_period),
        )
        runs_monitoring_cycle_tracker.initialize(db_session)
        last_update_time = runs_monitoring_cycle_tracker.get_window(db_session)
        now = datetime.datetime.now(datetime.timezone.utc)

        if config.alerts.mode == mlrun.common.schemas.alert.AlertsModes.enabled:
            _generate_event_on_failed_runs(db, db_session, last_update_time)
        _push_terminal_run_notifications(db, db_session, last_update_time)

        runs_monitoring_cycle_tracker.update_window(db_session, now)
    except Exception as exc:
        logger.warning(
            "Failed pushing terminal run notifications. Ignoring",
            exc=err_to_str(exc),
        )

    return stale_runs


def _cleanup_runtimes():
    db_session = create_session()
    try:
        for kind in RuntimeKinds.runtime_with_handlers():
            try:
                runtime_handler = get_runtime_handler(kind)
                runtime_handler.delete_resources(get_db(), db_session)
            except Exception as exc:
                logger.warning(
                    "Failed deleting resources. Ignoring",
                    exc=err_to_str(exc),
                    kind=kind,
                )
    finally:
        close_session(db_session)


def _push_terminal_run_notifications(
    db: server.api.db.base.DBInterface, db_session, last_update_time
):
    """
    Get all runs with notification configs which became terminal since the last call to the function
    and push their notifications if they haven't been pushed yet.
    """

    # When pushing notifications, push notifications only for runs that entered a terminal state
    # since the last time we pushed notifications.
    # On the first time we push notifications, we'll push notifications for all runs that are in a terminal state
    # and their notifications haven't been sent yet.

    runs = db.list_runs(
        db_session,
        project="*",
        states=mlrun.common.runtimes.constants.RunStates.terminal_states(),
        last_update_time_from=last_update_time,
        with_notifications=True,
    )

    if not len(runs):
        return

    # Unmasking the run parameters from secrets before handing them over to the notification handler
    # as importing the `Secrets` crud in the notification handler will cause a circular import
    unmasked_runs = [
        server.api.api.utils.unmask_notification_params_secret_on_task(
            db, db_session, run
        )
        for run in runs
    ]

    logger.debug(
        "Got terminal runs with configured notifications", runs_amount=len(runs)
    )
    server.api.utils.notification_pusher.RunNotificationPusher(unmasked_runs).push()


def _generate_event_on_failed_runs(
    db: server.api.db.base.DBInterface, db_session, last_update_time
):
    """
    Send an event on the runs that ended with error state since the last call to the function
    """

    runs = db.list_runs(
        db_session,
        project="*",
        states=[mlrun.common.runtimes.constants.RunStates.error],
        last_update_time_from=last_update_time,
    )

    for run in runs:
        project = run["metadata"]["project"]
        run_uid = run["metadata"]["uid"]
        run_name = run["metadata"]["name"]
        entity = mlrun.common.schemas.alert.EventEntities(
            kind=alert_objects.EventEntityKind.JOB,
            project=project,
            ids=[run_name],
        )
        event_value = {"uid": run_uid, "error": run["status"].get("error", None)}
        event_data = mlrun.common.schemas.Event(
            kind=alert_objects.EventKind.FAILED, entity=entity, value_dict=event_value
        )

        server.api.crud.Events().process_event(
            session=db_session,
            event_data=event_data,
            event_name=alert_objects.EventKind.FAILED,
            project=project,
            validate_event=True,
        )


async def _abort_stale_runs(stale_runs: list[dict]):
    semaphore = asyncio.Semaphore(
        int(mlrun.mlconf.monitoring.runs.concurrent_abort_stale_runs_workers)
    )

    async def abort_run(stale_run):
        # Using semaphore to limit the chunk we get from the thread pool for run aborting
        async with semaphore:
            # mark abort as internal, it doesn't have a background task
            stale_run["new_background_task_id"] = (
                server.api.constants.internal_abort_task_id
            )
            await fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                server.api.crud.Runs().abort_run,
                **stale_run,
            )

    coroutines = [abort_run(_stale_run) for _stale_run in stale_runs]
    if coroutines:
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Failed aborting stale run. Ignoring",
                    exc=err_to_str(result),
                )


async def _stop_logs():
    """
    Stop logs for runs that are in terminal state and last updated in the previous interval
    """
    logger.debug(
        "Getting all runs which reached terminal state in the previous interval and have logs requested",
        interval_seconds=int(config.log_collector.stop_logs_interval),
    )
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    try:
        runs = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[True],
            only_uids=False,
            states=mlrun.common.runtimes.constants.RunStates.terminal_states(),
            last_update_time_from=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=1.5 * config.log_collector.stop_logs_interval),
        )

        if len(runs) > 0:
            logger.debug(
                "Stopping logs for runs which reached terminal state in the previous interval",
                runs_count=len(runs),
            )
            await _stop_logs_for_runs(runs)
    finally:
        await fastapi.concurrency.run_in_threadpool(close_session, db_session)


async def _stop_logs_for_runs(runs: list, chunk_size: int = 10):
    project_to_run_uids = collections.defaultdict(list)
    for run in runs:
        project_name = run.get("metadata", {}).get("project", None)
        run_uid = run.get("metadata", {}).get("uid", None)
        project_to_run_uids[project_name].append(run_uid)

    for project_name, run_uids in project_to_run_uids.items():
        if not run_uids:
            logger.debug("No runs to stop logs for", project=project_name)
            continue

        # if we won't chunk the run uids, the grpc message might include many uids which will overflow the max message
        # size.
        for chunked_run_uids in mlrun.utils.helpers.iterate_list_by_chunks(
            run_uids, chunk_size
        ):
            try:
                await server.api.utils.clients.log_collector.LogCollectorClient().stop_logs(
                    project_name, chunked_run_uids
                )
            except Exception as exc:
                logger.warning(
                    "Failed stopping logs for runs. Ignoring",
                    exc=err_to_str(exc),
                    project=project_name,
                    chunked_run_uids=chunked_run_uids,
                )


if __name__ == "__main__":
    # this is for running the api server as part of
    # __main__.py on mlrun client and mlrun integration tests.
    # mlrun container image will run the server using uvicorn directly.
    # see /dockerfiles/mlrun-api/Dockerfile for more details.
    import server.api.apiuvicorn as uvicorn

    uvicorn.run(logger, httpdb_config=config.httpdb)
