# Copyright 2024 Iguazio
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
import datetime

import fastapi

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
from mlrun import mlconf

import framework.api.deps
import framework.constants
import framework.db.base
import framework.db.session
import framework.db.sqldb.db
import framework.service
import framework.utils.periodic
import framework.utils.singletons.db
import framework.utils.time_window_tracker
import services.alerts.crud
import services.alerts.initial_data
from framework.routers import alerts, auth, healthz


class Service(framework.service.Service):
    async def move_service_to_online(self):
        # TODO: Once alerts runs in its own pod - remove chief check
        if (
            mlconf.httpdb.clusterization.role
            == mlrun.common.schemas.ClusterizationRole.chief
        ):
            services.alerts.initial_data.update_default_configuration_data(self._logger)
            await self._start_periodic_functions()

    def _register_routes(self):
        # TODO: Resolve these dynamically from configuration
        alerts_v1_router = fastapi.APIRouter(
            dependencies=[fastapi.Depends(framework.api.deps.verify_api_state)]
        )
        alerts_v1_router.include_router(healthz.router, tags=["healthz"])
        alerts_v1_router.include_router(
            auth.router,
            tags=["auth"],
            dependencies=[fastapi.Depends(framework.api.deps.authenticate_request)],
        )
        alerts_v1_router.include_router(
            alerts.router,
            tags=["alerts"],
            dependencies=[fastapi.Depends(framework.api.deps.authenticate_request)],
        )
        self.app.include_router(
            alerts_v1_router, prefix=self.BASE_VERSIONED_SERVICE_PREFIX
        )

    async def _custom_setup_service(self):
        pass

    async def _start_periodic_functions(self):
        self._start_periodic_cleanup()

    def _start_periodic_cleanup(self):
        interval = int(mlconf.monitoring.runs.interval)
        if interval > 0:
            self._logger.info("Starting periodic runtimes cleanup", interval=interval)
            framework.utils.periodic.run_function_periodically(
                interval,
                self._monitor_runs_and_push_terminal_notifications.__name__,
                False,
                self._monitor_runs_and_push_terminal_notifications,
            )

    def _monitor_runs_and_push_terminal_notifications(self, db_session):
        db = framework.utils.singletons.db.get_db()
        try:
            runs_monitoring_cycle_tracker = framework.utils.time_window_tracker.TimeWindowTracker(
                key=framework.utils.time_window_tracker.TimeWindowTrackerKeys.run_monitoring,
                max_window_size_seconds=int(
                    mlconf.runtime_resources_deletion_grace_period
                ),
            )
            runs_monitoring_cycle_tracker.initialize(db_session)
            last_update_time = runs_monitoring_cycle_tracker.get_window(db_session)
            now = datetime.datetime.now(datetime.timezone.utc)

            self._generate_event_on_failed_runs(db, db_session, last_update_time)

            runs_monitoring_cycle_tracker.update_window(db_session, now)
        except Exception as exc:
            self._logger.warning(
                "Failed pushing terminal run notifications. Ignoring",
                exc=mlrun.errors.err_to_str(exc),
            )

    def _generate_event_on_failed_runs(
        self, db: framework.db.base.DBInterface, db_session, last_update_time
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
                kind=alert_objects.EventKind.FAILED,
                entity=entity,
                value_dict=event_value,
            )

            services.alerts.crud.Events().process_event(
                session=db_session,
                event_data=event_data,
                event_name=alert_objects.EventKind.FAILED,
                project=project,
                validate_event=True,
            )
