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
import datetime
import traceback
import typing

import fastapi
import fastapi.concurrency
import sqlalchemy.orm
import uvicorn
from fastapi.exception_handlers import http_exception_handler

import mlrun.api.db.base
import mlrun.api.utils.clients.chief
import mlrun.api.utils.clients.log_collector
import mlrun.common.schemas
import mlrun.errors
import mlrun.lists
import mlrun.utils
import mlrun.utils.notifications
import mlrun.utils.version
from mlrun.api.api.api import api_router
from mlrun.api.db.session import close_session, create_session
from mlrun.api.initial_data import init_data
from mlrun.api.middlewares import init_middlewares
from mlrun.api.runtime_handlers import get_runtime_handler
from mlrun.api.utils.periodic import (
    cancel_all_periodic_functions,
    cancel_periodic_function,
    run_function_periodically,
)
from mlrun.api.utils.singletons.db import get_db, initialize_db
from mlrun.api.utils.singletons.k8s import get_k8s_helper
from mlrun.api.utils.singletons.logs_dir import initialize_logs_dir
from mlrun.api.utils.singletons.project_member import (
    get_project_member,
    initialize_project_member,
)
from mlrun.api.utils.singletons.scheduler import get_scheduler, initialize_scheduler
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.runtimes import RuntimeClassMode, RuntimeKinds
from mlrun.utils import logger

API_PREFIX = "/api"
BASE_VERSIONED_API_PREFIX = f"{API_PREFIX}/v1"

# When pushing notifications, push notifications only for runs that entered a terminal state
# since the last time we pushed notifications.
# On the first time we push notifications, we'll push notifications for all runs that are in a terminal state
# and their notifications haven't been sent yet.
# TODO: find better solution than a global variable for chunking the list of runs
#      for which to push notifications
_last_notification_push_time: datetime.datetime = None

# This is a dictionary which holds the number of consecutive start log requests for each run uid.
# We use this dictionary to make sure that we don't get stuck in an endless loop of trying to collect logs for a runs
# that keep failing start logs requests.
_run_uid_start_log_request_counters: collections.Counter = collections.Counter()


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
# so new users won't use the old un-versioned api.
# /api points to /api/v1 since it is used externally, and we don't want to break it.
# TODO: make sure UI and all relevant Iguazio versions uses /api/v1 and deprecate this
app.include_router(api_router, prefix=API_PREFIX, include_in_schema=False)

init_middlewares(app)


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
    logger.warning(
        "Request handling returned error status",
        error_message=error_message,
        status_code=status_code,
        traceback=traceback.format_exc(),
        request_id=request_id,
    )
    return await http_exception_handler(
        request,
        fastapi.HTTPException(status_code=status_code, detail=error_message),
    )


@app.on_event("startup")
async def startup_event():
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

    if (
        config.httpdb.clusterization.worker.sync_with_chief.mode
        == mlrun.common.schemas.WaitForChiefToReachOnlineStateFeatureFlag.enabled
        and config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.worker
    ):
        _start_chief_clusterization_spec_sync_loop()

    if config.httpdb.state == mlrun.common.schemas.APIStates.online:
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
    if (
        config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.chief
    ):
        mlrun.api.initial_data.update_default_configuration_data()
        # runs cleanup/monitoring is not needed if we're not inside kubernetes cluster
        if get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
            _start_periodic_cleanup()
            _start_periodic_runs_monitoring()
            await _start_periodic_logs_collection()
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
    with the new log collection method and runs which might have reached terminal state while the API was down will
    have their logs collected as well.
    :param start_logs_limit: Semaphore which limits the number of concurrent log collection tasks
    """
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    logger.debug(
        "Getting all runs which are in non terminal state and require logs collection"
    )
    runs = await fastapi.concurrency.run_in_threadpool(
        get_db().list_distinct_runs_uids,
        db_session,
        requested_logs_modes=[None, False],
        only_uids=False,
        states=mlrun.runtimes.constants.RunStates.non_terminal_states(),
    )
    logger.debug(
        "Getting all runs which might have reached terminal state while the API was down",
        api_downtime_grace_period=config.log_collector.api_downtime_grace_period,
    )
    runs.extend(
        await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[None, False],
            only_uids=False,
            # We take the minimum between the api_downtime_grace_period and the runtime_resources_deletion_grace_period
            # because we want to make sure that we don't miss any runs which might have reached terminal state while the
            # API was down, and their runtime resources are not deleted
            last_update_time_from=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(
                seconds=min(
                    int(config.log_collector.api_downtime_grace_period),
                    int(config.runtime_resources_deletion_grace_period),
                )
            ),
            states=mlrun.runtimes.constants.RunStates.terminal_states(),
        )
    )
    if runs:
        logger.debug(
            "Found runs which require logs collection on startup",
            runs_uids=[run.get("metadata", {}).get("uid", None) for run in runs],
        )

        # we're using best_effort=True so the api will mark the runs as requested logs collection even in cases
        # where the log collection failed (e.g. when the pod is not found for runs that might have reached terminal
        # state while the API was down)
        await _start_log_and_update_runs(
            start_logs_limit, db_session, runs, best_effort=True
        )


async def _initiate_logs_collection(start_logs_limit: asyncio.Semaphore):
    """
    This function is responsible for initiating the logs collection process. It will get a list of all runs which are
    in a state which requires logs collection and will initiate the logs collection process for each of them.
    :param start_logs_limit: a semaphore which limits the number of concurrent logs collection processes
    """
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    try:
        # we don't want initiate logs collection for aborted runs
        run_states = mlrun.runtimes.constants.RunStates.all()
        run_states.remove(mlrun.runtimes.constants.RunStates.aborted)

        # list all the runs in the system which we didn't request logs collection for yet
        runs = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[False],
            only_uids=False,
            states=run_states,
        )
        if runs:
            logger.debug(
                "Found runs which require logs collection",
                runs_uids=[run.get("metadata", {}).get("uid", None) for run in runs],
            )
            await _start_log_and_update_runs(start_logs_limit, db_session, runs)

    finally:
        await fastapi.concurrency.run_in_threadpool(close_session, db_session)


async def _start_log_and_update_runs(
    start_logs_limit: asyncio.Semaphore,
    db_session: sqlalchemy.orm.Session,
    runs: list,
    best_effort: bool = False,
):
    if not runs:
        return

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
            mlrun.api.utils.clients.log_collector.LogCollectorClient()
        )
        run_kind = run.get("metadata", {}).get("labels", {}).get("kind", None)
        project_name = run.get("metadata", {}).get("project", None)
        run_uid = run.get("metadata", {}).get("uid", None)

        # information for why runtime isn't log collectable is inside the method
        if not mlrun.runtimes.RuntimeKinds.is_log_collectable_runtime(run_kind):
            # we mark the run as requested logs collection so we won't iterate over it again
            return run_uid
        try:
            runtime_handler: mlrun.api.runtime_handlers.BaseRuntimeHandler = (
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
    interval = int(config.runs_monitoring_interval)
    if interval > 0:
        logger.info("Starting periodic runs monitoring", interval=interval)
        run_function_periodically(
            interval, _monitor_runs.__name__, False, _monitor_runs
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
    Pulls runs from DB that are in terminal state and have logs requested, and call stop logs for them.
    This is done so that the log collector won't keep trying to collect logs for runs that are already
    in terminal state.
    """
    logger.debug(
        "Getting all runs which have reached terminal state and already have logs requested",
    )
    db_session = await fastapi.concurrency.run_in_threadpool(create_session)
    try:
        runs = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[True],
            only_uids=False,
            states=mlrun.runtimes.constants.RunStates.terminal_states()
            + [
                # add unknown state as well, as it's possible that the run reached such state
                # usually it happens when run pods get preempted
                mlrun.runtimes.constants.RunStates.unknown,
            ],
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
        chief_client = mlrun.api.utils.clients.chief.Client()
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


def _monitor_runs():
    db = get_db()
    db_session = create_session()
    try:
        for kind in RuntimeKinds.runtime_with_handlers():
            try:
                runtime_handler = get_runtime_handler(kind)
                runtime_handler.monitor_runs(db, db_session)
            except Exception as exc:
                logger.warning(
                    "Failed monitoring runs. Ignoring",
                    exc=err_to_str(exc),
                    kind=kind,
                )
        _push_terminal_run_notifications(db, db_session)
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
                    "Failed deleting resources. Ignoring",
                    exc=err_to_str(exc),
                    kind=kind,
                )
    finally:
        close_session(db_session)


def _push_terminal_run_notifications(db: mlrun.api.db.base.DBInterface, db_session):
    """
    Get all runs with notification configs which became terminal since the last call to the function
    and push their notifications if they haven't been pushed yet.
    """
    # Import here to avoid circular import
    import mlrun.api.api.utils

    # When pushing notifications, push notifications only for runs that entered a terminal state
    # since the last time we pushed notifications.
    # On the first time we push notifications, we'll push notifications for all runs that are in a terminal state
    # and their notifications haven't been sent yet.
    global _last_notification_push_time

    now = datetime.datetime.now(datetime.timezone.utc)

    runs = db.list_runs(
        db_session,
        project="*",
        states=mlrun.runtimes.constants.RunStates.terminal_states(),
        last_update_time_from=_last_notification_push_time,
        with_notifications=True,
    )

    if not len(runs):
        return

    # Unmasking the run parameters from secrets before handing them over to the notification handler
    # as importing the `Secrets` crud in the notification handler will cause a circular import
    unmasked_runs = [
        mlrun.api.api.utils.unmask_notification_params_secret_on_task(
            db, db_session, run
        )
        for run in runs
    ]

    logger.debug(
        "Got terminal runs with configured notifications", runs_amount=len(runs)
    )
    mlrun.utils.notifications.NotificationPusher(unmasked_runs).push()

    _last_notification_push_time = now


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
            states=mlrun.runtimes.constants.RunStates.terminal_states(),
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

        # if we wont chunk the run uids, the grpc message might include many uids which will overflow the max message
        # size.
        for chunked_run_uids in mlrun.utils.helpers.iterate_list_by_chunks(
            run_uids, chunk_size
        ):
            try:
                await mlrun.api.utils.clients.log_collector.LogCollectorClient().stop_logs(
                    project_name, chunked_run_uids
                )
            except Exception as exc:
                logger.warning(
                    "Failed stopping logs for runs. Ignoring",
                    exc=err_to_str(exc),
                    project=project_name,
                    chunked_run_uids=chunked_run_uids,
                )


def main():
    if (
        config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.chief
    ):
        init_data()
    elif (
        config.httpdb.clusterization.worker.sync_with_chief.mode
        == mlrun.common.schemas.WaitForChiefToReachOnlineStateFeatureFlag.enabled
        and config.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.worker
    ):
        # we set this state to mark the phase between the startup of the instance until we able to pull the chief state
        config.httpdb.state = mlrun.common.schemas.APIStates.waiting_for_chief

    logger.info(
        "Starting API server",
        port=config.httpdb.port,
        debug=config.httpdb.debug,
    )
    uvicorn.run(
        "mlrun.api.main:app",
        host="0.0.0.0",
        port=config.httpdb.port,
        access_log=False,
        timeout_keep_alive=config.httpdb.http_connection_timeout_keep_alive,
    )


if __name__ == "__main__":
    main()
