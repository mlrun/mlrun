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
import datetime
import traceback
import typing

import fastapi
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.lists
import mlrun.utils
import mlrun.utils.notifications
import mlrun.utils.version
from mlrun import mlconf
from mlrun.errors import err_to_str
from mlrun.runtimes import RuntimeClassMode, RuntimeKinds

import framework.api.utils
import framework.constants
import framework.db.base
import framework.service
import framework.utils.clients.chief
import framework.utils.clients.discovery
import framework.utils.clients.log_collector
import framework.utils.clients.messaging
import framework.utils.notifications.notification_pusher
import framework.utils.time_window_tracker
import services.api.crud
import services.api.initial_data
import services.api.runtime_handlers
import services.api.utils.db.partitioner
from framework.db.session import close_session, create_session
from framework.utils.periodic import (
    run_function_periodically,
)
from framework.utils.singletons.db import get_db
from framework.utils.singletons.k8s import get_k8s_helper
from framework.utils.singletons.project_member import (
    get_project_member,
    initialize_project_member,
)
from services.api.api.api import api_router, api_v2_router
from services.api.runtime_handlers import get_runtime_handler
from services.api.utils.singletons.logs_dir import initialize_logs_dir
from services.api.utils.singletons.scheduler import (
    ensure_scheduler,
    get_scheduler,
    start_scheduler,
)

# This is a dictionary which holds the number of consecutive start log requests for each run uid.
# We use this dictionary to make sure that we don't get stuck in an endless loop of trying to collect logs for a runs
# that keep failing start logs requests.
_run_uid_start_log_request_counters: collections.Counter = collections.Counter()


class Service(framework.service.Service):
    async def _move_service_to_online(self):
        # scheduler is needed on both workers and chief
        # on workers - it allows to us to list/get scheduler(s)
        # on chief - it allows to us to create/update/delete schedule(s)
        ensure_scheduler()
        if (
            mlconf.httpdb.clusterization.role
            == mlrun.common.schemas.ClusterizationRole.chief
            and mlconf.httpdb.clusterization.chief.feature_gates.scheduler == "enabled"
        ):
            await start_scheduler()

        # In general, it makes more sense to initialize the project member before the scheduler but in 1.1.0 in follower
        # we've added the full sync on the project member initialization (see code there for details) which might delete
        # projects which requires the scheduler to be set
        await fastapi.concurrency.run_in_threadpool(initialize_project_member)
        get_project_member().start()

        # maintenance periodic functions should only run on the chief instance
        if (
            mlconf.httpdb.clusterization.role
            == mlrun.common.schemas.ClusterizationRole.chief
        ):
            services.api.initial_data.update_default_configuration_data()
            await self._start_periodic_functions()

        await self._move_mounted_services_to_online()

    async def _base_handler(
        self,
        request: fastapi.Request,
        *args,
        **kwargs,
    ):
        messaging_client = framework.utils.clients.messaging.Client()
        return await messaging_client.proxy_request(request=request)

    def _register_routes(self):
        # TODO: This should be configurable and resolved in the base class
        self.app.include_router(api_router, prefix=self.base_versioned_service_prefix)
        self.app.include_router(api_v2_router, prefix=self.v2_service_prefix)
        # This is for backward compatibility, that is why we still leave it here but not include it in the schema
        # so new users won't use the old un-versioned api.
        # /api points to /api/v1 since it is used externally, and we don't want to break it.
        # TODO: make sure UI and all relevant Iguazio versions uses /api/v1 and deprecate this
        self.app.include_router(
            api_router, prefix=self.service_prefix, include_in_schema=False
        )

    async def _custom_setup_service(self):
        initialize_logs_dir()

    async def _custom_teardown_service(self):
        if get_project_member():
            get_project_member().shutdown()
        if get_scheduler():
            await get_scheduler().stop()

    def _initialize_data(self):
        if (
            mlconf.httpdb.clusterization.role
            == mlrun.common.schemas.ClusterizationRole.chief
        ):
            services.api.initial_data.init_data()

    async def _start_periodic_functions(self):
        # runs cleanup/monitoring is not needed if we're not inside kubernetes cluster
        if not get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
            return

        if mlconf.httpdb.clusterization.chief.feature_gates.cleanup == "enabled":
            self._start_periodic_cleanup()
        if (
            mlconf.httpdb.clusterization.chief.feature_gates.runs_monitoring
            == "enabled"
        ):
            self._start_periodic_runs_monitoring()
        if (
            mlconf.httpdb.clusterization.chief.feature_gates.pagination_cache
            == "enabled"
        ):
            self._start_periodic_pagination_cache_monitoring()
        if (
            mlconf.httpdb.clusterization.chief.feature_gates.project_summaries
            == "enabled"
        ):
            self._start_periodic_project_summaries_calculation()
        self._start_periodic_partition_management()
        self._start_periodic_refresh_smtp_configuration()
        if mlconf.httpdb.clusterization.chief.feature_gates.start_logs == "enabled":
            await self._start_periodic_logs_collection()
        if mlconf.httpdb.clusterization.chief.feature_gates.stop_logs == "enabled":
            await self._start_periodic_stop_logs()

    async def _start_periodic_logs_collection(
        self,
    ):
        if mlconf.log_collector.mode == mlrun.common.schemas.LogsCollectorMode.legacy:
            self._logger.info(
                "Using legacy logs collection method, skipping logs collection periodic function",
                mode=mlconf.log_collector.mode,
            )
            return
        self._logger.info(
            "Starting logs collection periodic function",
            mode=mlconf.log_collector.mode,
            interval=mlconf.log_collector.periodic_start_log_interval,
        )
        start_logs_limit = asyncio.Semaphore(
            mlconf.log_collector.concurrent_start_logs_workers
        )

        await self._verify_log_collection_started_on_startup(start_logs_limit)

        run_function_periodically(
            interval=int(mlconf.log_collector.periodic_start_log_interval),
            name=self._initiate_logs_collection.__name__,
            replace=False,
            function=self._initiate_logs_collection,
            start_logs_limit=start_logs_limit,
        )

    async def _verify_log_collection_started_on_startup(
        self,
        start_logs_limit: asyncio.Semaphore,
    ):
        """
        Verifies that log collection was started on startup for all runs which might have started before the API
        initialization or after upgrade.
        In that case we want to make sure that all runs which are in non-terminal state will have their logs collected
        by the log-collector and runs which might have reached terminal state while the API was down will have their
        logs collected as well.
        If the amount of runs which require logs collection on startup exceeds the configured limit, we will skip the
        rest but mark them as requested logs collection, to not get the API stuck in an endless loop of trying to
        collect logs.
        :param start_logs_limit: Semaphore which limits the number of concurrent log collection tasks
        """
        db_session = await fastapi.concurrency.run_in_threadpool(create_session)
        try:
            await framework.utils.time_window_tracker.run_with_time_window_tracker(
                db_session,
                key=framework.utils.time_window_tracker.TimeWindowTrackerKeys.log_collection,
                # If the API was down for more than the grace period, we will only collect logs for runs which reached
                # terminal state within the grace period and not since the API actually went down.
                max_window_size_seconds=min(
                    int(mlconf.log_collector.api_downtime_grace_period),
                    int(mlconf.runtime_resources_deletion_grace_period),
                ),
                ensure_window_update=True,
                callback=self._verify_log_collection_started,
                start_logs_limit=start_logs_limit,
            )
        finally:
            await fastapi.concurrency.run_in_threadpool(close_session, db_session)

    async def _verify_log_collection_started(
        self, db_session, last_update_time: datetime.datetime, start_logs_limit
    ):
        self._logger.debug(
            "Getting all runs which are in non terminal state and require logs collection"
        )
        runs_uids = await fastapi.concurrency.run_in_threadpool(
            get_db().list_distinct_runs_uids,
            db_session,
            requested_logs_modes=[None, False],
            only_uids=True,
            states=mlrun.common.runtimes.constants.RunStates.non_terminal_states(),
        )
        self._logger.debug(
            "Getting all runs which might have reached terminal state while the API was down",
            api_downtime_grace_period=mlconf.log_collector.api_downtime_grace_period,
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
                self._logger.warning(
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

            self._logger.debug(
                "Found runs which require logs collection on startup",
                runs_count=len(runs_uids),
            )

            # we're using best_effort=True so the api will mark the runs as requested logs collection even in cases
            # where the log collection failed (e.g. when the pod is not found for runs that might have reached
            # terminal state while the API was down)
            await self._start_log_and_update_runs(
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

    async def _initiate_logs_collection(self, start_logs_limit: asyncio.Semaphore):
        """
        This function is responsible for initiating the logs collection process. It will get a list of all runs which
        are in a state which requires logs collection and will initiate the logs collection process for each of them.
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
                seconds=int(mlconf.runtime_resources_deletion_grace_period)
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
                self._logger.debug(
                    "Found runs which require logs collection",
                    runs_uids=len(runs_uids),
                )
                await self._start_log_and_update_runs(
                    start_logs_limit=start_logs_limit,
                    db_session=db_session,
                    runs_uids=runs_uids,
                )

        finally:
            await fastapi.concurrency.run_in_threadpool(close_session, db_session)

    async def _start_log_and_update_runs(
        self,
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
            int(mlconf.log_collector.failed_runs_grace_period)
            / int(mlconf.log_collector.periodic_start_log_interval)
        )

        global _run_uid_start_log_request_counters
        runs_to_mark_as_requested_logs = []
        start_logs_for_runs = []
        for run in runs:
            run_uid = run.get("metadata", {}).get("uid", None)

            # if we requested logs for the same run more times than the threshold, we mark it as requested logs
            # collection, so the API and the log collector won't be stuck in an endless loop of trying to collect
            # logs for it
            if (
                run_uid in _run_uid_start_log_request_counters
                and _run_uid_start_log_request_counters[run_uid]
                >= max_consecutive_start_log_requests
            ):
                self._logger.warning(
                    "Run reached max consecutive start log requests, marking it as requested logs collection",
                    run_uid=run_uid,
                    requests_count=_run_uid_start_log_request_counters[run_uid],
                )
                runs_to_mark_as_requested_logs.append(run_uid)
                continue

            start_logs_for_runs.append(
                self._start_log_for_run(
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
            self._logger.debug(
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
        self,
        run: dict,
        start_logs_limit: typing.Optional[asyncio.Semaphore] = None,
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
                framework.utils.clients.log_collector.LogCollectorClient()
            )
            run_kind = run.get("metadata", {}).get("labels", {}).get("kind", None)
            project_name = run.get("metadata", {}).get("project", None)
            run_uid = run.get("metadata", {}).get("uid", None)

            # information for why runtime isn't log collectable is inside the method
            if not mlrun.runtimes.RuntimeKinds.is_log_collectable_runtime(run_kind):
                # we mark the run as requested logs collection so we won't iterate over it again
                return run_uid
            try:
                runtime_handler: services.api.runtime_handlers.BaseRuntimeHandler = (
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
                    # runtimes that the user will create with hundreds of resources (e.g mpi job can have multiple
                    # workers which aren't really important for log collection
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

                self._logger.warning(
                    "Failed to start logs for run",
                    run_uid=run_uid,
                    exc=mlrun.errors.err_to_str(exc),
                )
                return None

    def _start_periodic_cleanup(self):
        interval = int(mlconf.runtimes_cleanup_interval)
        if interval > 0:
            self._logger.info("Starting periodic runtimes cleanup", interval=interval)
            run_function_periodically(
                interval, self._cleanup_runtimes.__name__, False, self._cleanup_runtimes
            )

    def _start_periodic_runs_monitoring(self):
        interval = int(mlconf.monitoring.runs.interval)
        if interval > 0:
            self._logger.info("Starting periodic runs monitoring", interval=interval)
            run_function_periodically(
                interval, self._monitor_runs.__name__, False, self._monitor_runs
            )

    def _start_periodic_pagination_cache_monitoring(self):
        interval = int(mlconf.httpdb.pagination.pagination_cache.interval)
        if interval > 0:
            self._logger.info(
                "Starting periodic pagination cache monitoring", interval=interval
            )
            run_function_periodically(
                interval,
                services.api.crud.pagination_cache.PaginationCache().monitor_pagination_cache.__name__,
                False,
                framework.db.session.run_function_with_new_db_session,
                services.api.crud.pagination_cache.PaginationCache().monitor_pagination_cache,
            )

    def _start_periodic_project_summaries_calculation(self):
        interval = int(mlconf.monitoring.projects.summaries.cache_interval)
        if interval > 0:
            self._logger.info(
                "Starting periodic project summaries calculation", interval=interval
            )
            run_function_periodically(
                interval,
                services.api.crud.projects.Projects().refresh_project_resources_counters_cache.__name__,
                False,
                framework.db.session.run_async_function_with_new_db_session,
                services.api.crud.projects.Projects().refresh_project_resources_counters_cache,
            )

    def _start_periodic_partition_management(self):
        for table_name, retention_days in mlconf.object_retentions.items():
            self._logger.info(
                f"Starting periodic partition management for table {table_name}",
                retention_days=retention_days,
            )
            partition_interval = framework.db.session.run_function_with_new_db_session(
                services.api.utils.db.partitioner.MySQLPartitioner().get_partition_interval,
                table_name=table_name,
            )
            interval_in_seconds = int(
                partition_interval.as_duration().total_seconds() / 2
            )
            run_function_periodically(
                interval_in_seconds,
                f"{self._manage_partitions.__name__}_{table_name}",
                False,
                self._manage_partitions,
                table_name=table_name,
                retention_days=retention_days,
            )

    def _start_periodic_refresh_smtp_configuration(self):
        interval = int(mlconf.notifications.smtp.refresh_interval)
        if interval > 0:
            self._logger.info(
                "Starting periodic refresh SMTP configuration", interval=interval
            )
            run_function_periodically(
                interval,
                framework.utils.notifications.notification_pusher.RunNotificationPusher.get_mail_notification_default_params.__name__,
                False,
                framework.utils.notifications.notification_pusher.RunNotificationPusher.get_mail_notification_default_params,
                refresh=True,
            )

    @staticmethod
    async def _manage_partitions(table_name, retention_days):
        await fastapi.concurrency.run_in_threadpool(
            framework.db.session.run_function_with_new_db_session,
            services.api.utils.db.partitioner.MySQLPartitioner().create_and_drop_partitions,
            table_name=table_name,
            retention_days=retention_days,
        )

    async def _start_periodic_stop_logs(
        self,
    ):
        if mlconf.log_collector.mode == mlrun.common.schemas.LogsCollectorMode.legacy:
            self._logger.info(
                "Using legacy logs collection method, skipping stop logs periodic function",
                mode=mlconf.log_collector.mode,
            )
            return

        await self._verify_log_collection_stopped_on_startup()

        interval = int(mlconf.log_collector.stop_logs_interval)
        if interval > 0:
            self._logger.info("Starting periodic stop logs", interval=interval)
            run_function_periodically(
                interval, self._stop_logs.__name__, False, self._stop_logs
            )

    async def _verify_log_collection_stopped_on_startup(
        self,
    ):
        """
        First, list runs that are currently being collected in the log collector.
        Second, query the DB for those runs that are also in terminal state and have logs requested.
        Lastly, call stop logs for the runs that met all of the above conditions.
        This is done so that the log collector won't keep trying to collect logs for runs that are already
        in terminal state.
        """
        self._logger.debug("Listing runs currently being log collected")
        log_collector_client = (
            framework.utils.clients.log_collector.LogCollectorClient()
        )
        run_uids_in_progress = []
        failed_listing = False
        try:
            runs_in_progress_response_stream = (
                log_collector_client.list_runs_in_progress()
            )
            # collate the run uids from the response stream to a list
            async for run_uids in runs_in_progress_response_stream:
                run_uids_in_progress.extend(run_uids)
        except Exception as exc:
            failed_listing = True
            self._logger.warning(
                "Failed listing runs currently being log collected",
                exc=err_to_str(exc),
                traceback=traceback.format_exc(),
            )

        if len(run_uids_in_progress) == 0 and not failed_listing:
            self._logger.debug("No runs currently being log collected")
            return

        self._logger.debug(
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
                self._logger.debug(
                    "Stopping logs for runs which reached terminal state before startup",
                    runs_count=len(runs),
                )
                await self._stop_logs_for_runs(runs)
        finally:
            await fastapi.concurrency.run_in_threadpool(close_session, db_session)

    async def _monitor_runs(self):
        stale_runs = await framework.db.session.run_async_function_with_new_db_session(
            self._monitor_runs_and_push_terminal_notifications
        )
        await self._abort_stale_runs(stale_runs)

    async def _monitor_runs_and_push_terminal_notifications(self, db_session):
        db = get_db()
        stale_runs = []
        for kind in RuntimeKinds.runtime_with_handlers():
            try:
                runtime_handler = get_runtime_handler(kind)
                runtime_stale_runs = runtime_handler.monitor_runs(db, db_session)
                stale_runs.extend(runtime_stale_runs)
            except Exception as exc:
                self._logger.warning(
                    "Failed monitoring runs. Ignoring",
                    exc=err_to_str(exc),
                    kind=kind,
                )
        try:
            await framework.utils.time_window_tracker.run_with_time_window_tracker(
                db_session,
                key=framework.utils.time_window_tracker.TimeWindowTrackerKeys.run_monitoring,
                max_window_size_seconds=int(
                    mlconf.runtime_resources_deletion_grace_period
                ),
                ensure_window_update=False,
                callback=self._push_terminal_run_notifications,
                db=db,
            )
        except Exception as exc:
            self._logger.warning(
                "Failed pushing terminal run notifications. Ignoring",
                exc=err_to_str(exc),
                traceback=traceback.format_exc(),
            )

        return stale_runs

    def _cleanup_runtimes(self):
        db_session = create_session()
        try:
            for kind in RuntimeKinds.runtime_with_handlers():
                try:
                    runtime_handler = get_runtime_handler(kind)
                    runtime_handler.delete_resources(get_db(), db_session)
                except Exception as exc:
                    self._logger.warning(
                        "Failed deleting resources. Ignoring",
                        exc=err_to_str(exc),
                        kind=kind,
                    )
        finally:
            close_session(db_session)

    def _push_terminal_run_notifications(
        self,
        db_session,
        last_update_time: datetime.datetime,
        db: framework.db.base.DBInterface,
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
        unmasked_runs = []
        for run in runs:
            try:
                framework.utils.notifications.unmask_notification_params_secret_on_task(
                    db, db_session, run
                )
                unmasked_runs.append(run)
            except Exception as exc:
                self._logger.warning(
                    "Failed unmasking notification params secret. Ignoring",
                    project=run.metadata.project,
                    run_uid=run.metadata.uid,
                    exc=err_to_str(exc),
                )

        self._logger.debug(
            "Got terminal runs with configured notifications", runs_amount=len(runs)
        )
        run_notification_pusher_class = (
            framework.utils.notifications.notification_pusher.RunNotificationPusher
        )
        run_notification_pusher_class(
            unmasked_runs,
            run_notification_pusher_class.resolve_notifications_default_params(),
        ).push()

    async def _abort_stale_runs(self, stale_runs: list[dict]):
        semaphore = asyncio.Semaphore(
            int(mlrun.mlconf.monitoring.runs.concurrent_abort_stale_runs_workers)
        )

        async def abort_run(stale_run):
            # Using semaphore to limit the chunk we get from the thread pool for run aborting
            async with semaphore:
                # mark abort as internal, it doesn't have a background task
                stale_run["new_background_task_id"] = (
                    framework.constants.internal_abort_task_id
                )
                await fastapi.concurrency.run_in_threadpool(
                    framework.db.session.run_function_with_new_db_session,
                    services.api.crud.Runs().abort_run,
                    **stale_run,
                )

        coroutines = [abort_run(_stale_run) for _stale_run in stale_runs]
        if coroutines:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self._logger.warning(
                        "Failed aborting stale run. Ignoring",
                        exc=err_to_str(result),
                    )

    async def _stop_logs(
        self,
    ):
        """
        Stop logs for runs that are in terminal state and last updated in the previous interval
        """
        self._logger.debug(
            "Getting all runs which reached terminal state in the previous interval and have logs requested",
            interval_seconds=int(mlconf.log_collector.stop_logs_interval),
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
                - datetime.timedelta(
                    seconds=1.5 * mlconf.log_collector.stop_logs_interval
                ),
            )

            if len(runs) > 0:
                self._logger.debug(
                    "Stopping logs for runs which reached terminal state in the previous interval",
                    runs_count=len(runs),
                )
                await self._stop_logs_for_runs(runs)
        finally:
            await fastapi.concurrency.run_in_threadpool(close_session, db_session)

    async def _stop_logs_for_runs(self, runs: list, chunk_size: int = 10):
        project_to_run_uids = collections.defaultdict(list)
        for run in runs:
            project_name = run.get("metadata", {}).get("project", None)
            run_uid = run.get("metadata", {}).get("uid", None)
            project_to_run_uids[project_name].append(run_uid)

        for project_name, run_uids in project_to_run_uids.items():
            if not run_uids:
                self._logger.debug("No runs to stop logs for", project=project_name)
                continue

            # if we won't chunk the run uids, the grpc message might include many uids which will overflow
            # the max message size.
            for chunked_run_uids in mlrun.utils.helpers.iterate_list_by_chunks(
                run_uids, chunk_size
            ):
                try:
                    await framework.utils.clients.log_collector.LogCollectorClient().stop_logs(
                        project_name, chunked_run_uids
                    )
                except Exception as exc:
                    self._logger.warning(
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
    import framework.utils.mlrunuvicorn as uvicorn

    uvicorn.run(httpdb_config=mlconf.httpdb, service_name="api")
