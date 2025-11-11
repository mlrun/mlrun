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

import unittest.mock
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from kubernetes import client as k8s_client
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import tests.conftest
from mlrun.common.runtimes.constants import PodPhases, RunStates
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.utils import now_date

import framework.utils.helpers
import framework.utils.singletons.db
import services.api.crud
from framework.utils.singletons.db import get_db
from services.api.runtime_handlers import get_runtime_handler
from services.api.tests.unit.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.kind = self._get_class_name()
        self.runtime_handler = get_runtime_handler(self._get_class_name())
        self.runtime_handler.wait_for_deletion_interval = 0

        self.job_labels = {
            mlrun_constants.MLRunInternalLabels.mlrun_class: self._get_class_name(),
            mlrun_constants.MLRunInternalLabels.function: "my-trainer",
            mlrun_constants.MLRunInternalLabels.name: "my-training",
            mlrun_constants.MLRunInternalLabels.project: self.project,
            mlrun_constants.MLRunInternalLabels.scrape_metrics: "False",
            mlrun_constants.MLRunInternalLabels.tag: "latest",
            mlrun_constants.MLRunInternalLabels.uid: self.run_uid,
        }
        job_pod_name = "my-training-j7dtf"

        # initializing them here to save space in tests
        self.pending_job_pod = self._generate_pod(
            job_pod_name, self.job_labels, PodPhases.pending
        )
        self.running_job_pod = self._generate_pod(
            job_pod_name, self.job_labels, PodPhases.running
        )
        self.completed_job_pod = self._generate_pod(
            job_pod_name, self.job_labels, PodPhases.succeeded
        )
        self.failed_job_pod = self._generate_pod(
            job_pod_name, self.job_labels, PodPhases.failed
        )

        builder_legacy_labels = {
            mlrun_constants.MLRunInternalLabels.mlrun_class: "build",
            mlrun_constants.MLRunInternalLabels.task_name: "mlrun-build-hedi-simple-func-legacy",
        }
        builder_legacy_pod_name = "mlrun-build-hedi-simple-legacy-func-8qwrd"
        self.completed_legacy_builder_pod = self._generate_pod(
            builder_legacy_pod_name, builder_legacy_labels, PodPhases.succeeded
        )

    def _get_class_name(self):
        return "job"

    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, expected_pods=pods
        )

    def test_list_resources_grouped_by(self, db: Session, client: TestClient):
        for group_by in [
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.job,
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.project,
        ]:
            pods = self._mock_list_resources_pods()
            self._assert_runtime_handler_list_resources(
                RuntimeKinds.job,
                expected_pods=pods,
                group_by=group_by,
            )

    def test_list_resources_grouped_by_project_with_non_project_resources(
        self, db: Session, client: TestClient
    ):
        pods = self._mock_list_resources_pods(self.completed_legacy_builder_pod)
        resources = self._assert_runtime_handler_list_resources(
            RuntimeKinds.job,
            expected_pods=pods,
            group_by=mlrun.common.schemas.ListRuntimeResourcesGroupByField.project,
        )
        # the legacy builder pod does not have a project label, verify it is listed under the empty key
        # so it will be removed on cleanup
        assert "" in resources

    @pytest.mark.asyncio
    async def test_delete_resources_completed_pod(
        self, db: Session, client: TestClient
    ):
        list_namespaced_pods_calls = [
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
            # additional time for wait for pods deletion - simulate pod not removed yet
            [self.completed_job_pod],
            # additional time for wait for pods deletion - simulate pod gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods(
            [self.completed_job_pod.metadata.name],
            self.completed_job_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed, requested_logs=True
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_job_pod.metadata.name,
        )

    def test_delete_resources_completed_builder_pod(
        self, db: Session, client: TestClient
    ):
        """
        Test mainly used to verify that we're not spamming errors in logs in this specific scenario
        """
        list_namespaced_pods_calls = [
            [self.completed_legacy_builder_pod],
            # additional time for the get_logger_pods
            [self.completed_legacy_builder_pod],
            # additional time for wait for pods deletion - simulate pod not removed yet
            [self.completed_legacy_builder_pod],
            # additional time for wait for pods deletion - simulate pod gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods(
            [self.completed_legacy_builder_pod.metadata.name],
            self.completed_legacy_builder_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_running_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)

        # nothing removed cause pod is running
        self._assert_delete_namespaced_pods([])
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_with_grace_period(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10)

        # nothing removed cause pod grace period didn't pass
        self._assert_delete_namespaced_pods([])
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    @pytest.mark.parametrize(
        "runs_monitoring_interval, debouncing_interval, list_namespaced_pods_calls, "
        "interval_time_to_add_to_run_update_time, start_run_states, expected_reached_state, monitor_cycles",
        [
            # no monitoring interval and no debouncing interval which means if run found in non-terminal state
            # the monitoring will override to terminal status
            (
                0,
                None,
                [[], []],
                0,
                RunStates.non_terminal_states(),
                RunStates.error,
                1,
            ),
            # monitoring interval and debouncing interval are configured which means debouncing interval will
            # be the debounce period, run is still in the debounce period that's why expecting not to override state
            # to terminal state
            (
                30,
                100,
                [[], [], []],
                -70,
                RunStates.non_terminal_states(),
                RunStates.non_terminal_states(),
                None,
            ),
            # monitoring interval and debouncing interval are configured which means debouncing interval will
            # be the debounce period, run update time passed the debounce period that's why expecting to override state
            # to terminal state
            (
                30,
                100,
                [[], [], [], []],
                -200,
                RunStates.non_terminal_states(),
                RunStates.error,
                3,
            ),
            # monitoring interval configured and debouncing interval isn't configured which means
            # monitoring interval * 2 will be the debounce period.
            # run isn't in the debounce period that's why expecting to override state to terminal state
            (
                30,
                None,
                [[], [], [], []],
                -65,
                RunStates.non_terminal_states(),
                RunStates.error,
                3,
            ),
            # monitoring interval configured and debouncing interval isn't configured which means
            # monitoring interval * 2 will be the debounce period.
            # run is in the debounce period that's why expecting not to override state to terminal state
            (
                30,
                None,
                [[], [], []],
                -35,
                RunStates.non_terminal_states(),
                RunStates.non_terminal_states(),
                None,
            ),
        ],
    )
    def test_ensure_run_not_stuck_on_non_terminal_state(
        self,
        db: Session,
        client: TestClient,
        runs_monitoring_interval,
        debouncing_interval,
        list_namespaced_pods_calls,
        interval_time_to_add_to_run_update_time,
        start_run_states,
        expected_reached_state,
        monitor_cycles,
    ):
        config.monitoring.runs.interval = runs_monitoring_interval or 0
        config.monitoring.runs.missing_runtime_resources_debouncing_interval = (
            debouncing_interval
        )
        monitor_cycles = monitor_cycles or len(list_namespaced_pods_calls)
        for idx in range(len(start_run_states)):
            self.run["status"]["state"] = start_run_states[idx]

            # using freeze enables us to set the now attribute when calling the sub-function
            # _update_run_updated_time without the need to call the function directly
            original_update_run_updated_time = (
                framework.utils.singletons.db.get_db()._update_run_updated_time
            )
            framework.utils.singletons.db.get_db()._update_run_updated_time = (
                tests.conftest.freeze(
                    original_update_run_updated_time,
                    now=now_date()
                    + timedelta(
                        seconds=interval_time_to_add_to_run_update_time,
                    ),
                )
            )
            services.api.crud.Runs().store_run(
                db, self.run, self.run_uid, project=self.project
            )
            framework.utils.singletons.db.get_db()._update_run_updated_time = (
                original_update_run_updated_time
            )
            # Mocking pod that is still in non-terminal state
            self._mock_list_namespaced_pods(list_namespaced_pods_calls)
            # using freeze enables us to set the now attribute when calling the sub-function
            # _update_run_updated_time without the need to call the function directly
            original_update_run_updated_time = (
                framework.utils.singletons.db.get_db()._update_run_updated_time
            )
            framework.utils.singletons.db.get_db()._update_run_updated_time = (
                tests.conftest.freeze(
                    original_update_run_updated_time,
                    now=now_date()
                    + timedelta(
                        seconds=interval_time_to_add_to_run_update_time,
                    ),
                )
            )
            services.api.crud.Runs().store_run(
                db, self.run, self.run_uid, project=self.project
            )
            framework.utils.singletons.db.get_db()._update_run_updated_time = (
                original_update_run_updated_time
            )
            # Mocking pod that is still in non-terminal state
            self._mock_list_namespaced_pods(list_namespaced_pods_calls)

            # Triggering monitor cycle
            for i in range(monitor_cycles):
                self.runtime_handler.monitor_runs(get_db(), db)

            expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
            self._assert_list_namespaced_pods_calls(
                self.runtime_handler, expected_number_of_list_pods_calls
            )

            # verifying monitoring was debounced
            if isinstance(expected_reached_state, list):
                self._assert_run_reached_state(
                    db, self.project, self.run_uid, expected_reached_state[idx]
                )
            else:
                self._assert_run_reached_state(
                    db, self.project, self.run_uid, expected_reached_state
                )
            get_db().del_run(db, self.run_uid, self.project)

    @pytest.mark.asyncio
    async def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_job_pod],
            # additional time for the get_logger_pods
            [self.running_job_pod],
            # additional time for wait for pods deletion - simulate pod gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10, force=True)
        self._assert_delete_namespaced_pods(
            [self.running_job_pod.metadata.name],
            self.running_job_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.running_job_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed, requested_logs=True
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_job_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_failed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            [self.failed_job_pod],
            # additional time for the get_logger_pods
            [self.failed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db,
            self.project,
            self.run_uid,
            RunStates.error,
            expected_status_attrs={
                "reason": "Some reason",
                "status_text": "Failed message",
            },
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.failed_job_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_debouncing_non_terminal_state(
        self, db: Session, client: TestClient
    ):
        # This test verifies that a run in a non-terminal state is not updated if it was already updated recently
        # (i.e., within the debounce interval).
        # It ensures the debounce logic correctly skips redundant updates for active runs.

        # set monitoring interval so debouncing will be active
        config.monitoring.runs.interval = 100

        # Mocking the SDK updating the Run's state to terminal state
        self.run["status"]["state"] = RunStates.completed
        original_update_run_updated_time = (
            framework.utils.singletons.db.get_db()._update_run_updated_time
        )
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            tests.conftest.freeze(original_update_run_updated_time, now=now_date())
        )
        services.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            original_update_run_updated_time
        )

        # Mocking pod that is still in non-terminal state
        self._mock_list_namespaced_pods([[self.running_job_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

        # Mocking that update occurred before debounced period
        debounce_period = config.monitoring.runs.interval
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            tests.conftest.freeze(
                original_update_run_updated_time,
                now=now_date() - timedelta(seconds=float(2 * debounce_period)),
            )
        )
        services.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            original_update_run_updated_time
        )

        # Mocking pod that is still in non-terminal state
        self._mock_list_namespaced_pods([[self.running_job_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

    @pytest.mark.asyncio
    async def test_monitor_run_debouncing_terminal_state(
        self, db: Session, client: TestClient
    ):
        # This test verifies the debounce logic when the runtime has reached a terminal state but the DB still shows a
        # recent non-terminal update. Initially, the update should be debounced.

        # Set monitoring interval so debouncing will be active
        config.monitoring.runs.interval = 100

        # Simulate record still in non-terminal state ("running")
        self.run["status"]["state"] = RunStates.running
        original_update_run_updated_time = (
            framework.utils.singletons.db.get_db()._update_run_updated_time
        )
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            tests.conftest.freeze(original_update_run_updated_time, now=now_date())
        )
        services.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            original_update_run_updated_time
        )

        # Simulate runtime already in terminal state (extra one for the log collection)
        self._mock_list_namespaced_pods([[self.running_job_pod]])

        # Trigger monitoring - this should be debounced and not overwrite DB "running"
        self.runtime_handler.monitor_runs(get_db(), db)

        # Verify that debounce happened: state in DB should still be "running"
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

        # Now simulate that debounce window has passed (simulate old update)
        debounce_period = config.monitoring.runs.interval
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            tests.conftest.freeze(
                original_update_run_updated_time,
                now=now_date() - timedelta(seconds=2 * debounce_period),
            )
        )
        services.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        framework.utils.singletons.db.get_db()._update_run_updated_time = (
            original_update_run_updated_time
        )

        # Mocking pod that is in terminal state (extra one for the log collection)
        self._mock_list_namespaced_pods(
            [[self.completed_job_pod], [self.completed_job_pod]]
        )

        # Mocking read log calls
        log = self._mock_read_namespaced_pod_log()

        # Re-run monitor (now update should go through)
        self.runtime_handler.monitor_runs(get_db(), db)

        # DB should now reflect the terminal state
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_job_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_run_does_not_exists(
        self, db: Session, client: TestClient
    ):
        get_db().del_run(db, self.run_uid, self.project)
        list_namespaced_pods_calls = [
            [self.running_job_pod],
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_job_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_state_thresholds_defaults(self, db: Session, client: TestClient):
        """
        Test that the default state thresholds are applied correctly
        This test creates 6 pods:
        - pending pod that is not scheduled - should not be deleted
        - running pod with new start time - should not be deleted
        - pending scheduled pod with new start time - should not be deleted
        - pending scheduled pod with old start time - should be deleted
        - running pod with old start time - should be deleted
        - pod in image pull backoff with old start time - should be deleted
        """
        pending_scheduled_labels = self._generate_job_labels(
            "pending_scheduled", job_labels=self.job_labels
        )
        pending_scheduled_pod = self._generate_pod(
            pending_scheduled_labels[mlrun_constants.MLRunInternalLabels.name],
            pending_scheduled_labels,
            PodPhases.pending,
        )
        pending_scheduled_pod.status.conditions = [
            k8s_client.V1PodCondition(type="PodScheduled", status="True")
        ]
        pending_scheduled_pod.status.start_time = datetime.now(
            timezone.utc
        ) - timedelta(
            seconds=framework.utils.helpers.time_string_to_seconds(
                mlrun.mlconf.function.spec.state_thresholds.default.pending_scheduled
            )
        )
        self._store_run(
            db,
            pending_scheduled_labels[mlrun_constants.MLRunInternalLabels.name],
            pending_scheduled_labels[mlrun_constants.MLRunInternalLabels.uid],
            start_time=pending_scheduled_pod.status.start_time,
        )

        pending_scheduled_new_labels = self._generate_job_labels(
            "pending_scheduled_new", job_labels=self.job_labels
        )
        pending_scheduled_pod_new = self._generate_pod(
            pending_scheduled_new_labels[mlrun_constants.MLRunInternalLabels.name],
            pending_scheduled_new_labels,
            PodPhases.pending,
        )
        pending_scheduled_pod_new.status.conditions = [
            k8s_client.V1PodCondition(type="PodScheduled", status="True")
        ]
        self._store_run(
            db,
            pending_scheduled_new_labels[mlrun_constants.MLRunInternalLabels.name],
            pending_scheduled_new_labels[mlrun_constants.MLRunInternalLabels.uid],
            start_time=pending_scheduled_pod_new.status.start_time,
        )

        running_overtime_labels = self._generate_job_labels(
            "running_overtime", job_labels=self.job_labels
        )
        running_overtime_pod = self._generate_pod(
            running_overtime_labels[mlrun_constants.MLRunInternalLabels.name],
            running_overtime_labels,
            PodPhases.running,
        )
        running_overtime_pod.status.start_time = datetime.now(timezone.utc) - timedelta(
            seconds=framework.utils.helpers.time_string_to_seconds(
                mlrun.mlconf.function.spec.state_thresholds.default.executing
            )
        )
        self._store_run(
            db,
            running_overtime_labels[mlrun_constants.MLRunInternalLabels.name],
            running_overtime_labels[mlrun_constants.MLRunInternalLabels.uid],
            start_time=running_overtime_pod.status.start_time,
        )

        image_pull_backoff_labels = self._generate_job_labels(
            "image_pull_backoff", job_labels=self.job_labels
        )
        image_pull_backoff_pod = self._generate_pod(
            image_pull_backoff_labels[mlrun_constants.MLRunInternalLabels.name],
            image_pull_backoff_labels,
            PodPhases.pending,
        )
        image_pull_backoff_pod.status.container_statuses = [
            k8s_client.V1ContainerStatus(
                image="some-image",
                image_id="some-image-id",
                name="some-container",
                ready=False,
                restart_count=10,
                state=k8s_client.V1ContainerState(
                    waiting=k8s_client.V1ContainerStateWaiting(
                        reason="ImagePullBackOff"
                    )
                ),
            )
        ]
        image_pull_backoff_pod.status.start_time = datetime.now(
            timezone.utc
        ) - timedelta(
            seconds=framework.utils.helpers.time_string_to_seconds(
                mlrun.mlconf.function.spec.state_thresholds.default.image_pull_backoff
            )
        )
        self._store_run(
            db,
            image_pull_backoff_labels[mlrun_constants.MLRunInternalLabels.name],
            image_pull_backoff_labels[mlrun_constants.MLRunInternalLabels.uid],
            start_time=image_pull_backoff_pod.status.start_time,
        )

        list_namespaced_pods_calls = [
            [
                self.pending_job_pod,
                self.running_job_pod,
                pending_scheduled_pod_new,
                pending_scheduled_pod,
                running_overtime_pod,
                image_pull_backoff_pod,
            ],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        stale_runs = self.runtime_handler.monitor_runs(get_db(), db)
        assert len(stale_runs) == 3

        stale_run_uids = [run["uid"] for run in stale_runs]
        expected_stale_run_uids = [
            pending_scheduled_pod.metadata.labels[
                mlrun_constants.MLRunInternalLabels.uid
            ],
            running_overtime_pod.metadata.labels[
                mlrun_constants.MLRunInternalLabels.uid
            ],
            image_pull_backoff_pod.metadata.labels[
                mlrun_constants.MLRunInternalLabels.uid
            ],
        ]
        assert stale_run_uids == expected_stale_run_uids

        stale_run_updates = [run["run_updates"] for run in stale_runs]
        expected_run_updates = []
        for state in ["pending_scheduled", "executing", "image_pull_backoff"]:
            expected_run_updates.append(
                {
                    "status.error": f"Run aborted due to exceeded state threshold: {state}",
                }
            )
        assert stale_run_updates == expected_run_updates

    @pytest.mark.asyncio
    async def test_monitor_stale_run(self, db: Session, client: TestClient):
        # set list run time period to be negative so that list runs will not find the run
        config.monitoring.runs.list_runs_time_period_in_days = -1
        list_namespaced_pods_calls = [
            [self.running_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_read_namespaced_pod_log()
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls
        )

        run = get_db().read_run(db, self.run_uid, self.project)
        with unittest.mock.patch(
            "framework.db.sqldb.db.SQLDB.read_run",
            unittest.mock.Mock(return_value=run),
        ) as mock_read_run:
            for _ in range(expected_monitor_cycles_to_reach_expected_state):
                self.runtime_handler.monitor_runs(get_db(), db)

            mock_read_run.assert_called_once()
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

    @pytest.mark.asyncio
    async def test_monitor_no_search_run(self, db: Session, client: TestClient):
        # tests the opposite of test_monitor_stale_run - that the run is listed, and we don't try to read it
        list_namespaced_pods_calls = [
            [self.completed_job_pod],
            # additional time for the get_logger_pods
            [self.completed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_read_namespaced_pod_log()
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )

        with unittest.mock.patch(
            "framework.db.sqldb.db.SQLDB.read_run", unittest.mock.Mock()
        ) as mock_read_run:
            for _ in range(expected_monitor_cycles_to_reach_expected_state):
                self.runtime_handler.monitor_runs(get_db(), db)

            mock_read_run.assert_not_called()
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

    @pytest.mark.asyncio
    async def test_monitor_run_debouncing_resource_not_found(
        self, db: Session, client: TestClient
    ):
        config.monitoring.runs.missing_runtime_resources_debouncing_interval = 0
        self.run["status"]["state"] = RunStates.running

        services.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )

        # Mocking once that the pod is not found, and then that it is found
        list_namespaced_pods_calls = [[], [self.completed_job_pod]]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    @pytest.mark.asyncio
    async def test_monitor_run_retry(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            [self.failed_job_pod],
            # additional time for the get_logger_pods
            [self.failed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        self._store_run(
            db,
            retry_spec={
                "count": 3,
            },
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db,
            self.project,
            self.run_uid,
            RunStates.pending_retry,
            expected_status_attrs={
                "reason": "Some reason",
                "status_text": "Run failed attempt 1 of 4 with error: Failed message",
            },
        )

    @pytest.mark.asyncio
    async def test_monitor_run_retry_exhausted(self, db: Session, client: TestClient):
        # label the pods with the retry attempt (3). Without this, the pods would remain unlabeled and the monitor
        # logic would treat them as outdated, causing them to be skipped.
        for pod in [self.pending_job_pod, self.running_job_pod, self.failed_job_pod]:
            pod.metadata.labels[mlrun.common.constants.MLRunInternalLabels.retry] = "3"

        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            [self.failed_job_pod],
            # additional time for the get_logger_pods
            [self.failed_job_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        self._store_run(
            db,
            retry_spec={
                "count": 3,
            },
            retry_count=3,
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db,
            self.project,
            self.run_uid,
            RunStates.error,
            expected_status_attrs={
                "reason": "Some reason",
                "status_text": "Run failed after 4 attempts with error: Failed message",
            },
        )

    @pytest.mark.asyncio
    async def test_retry_pod_deleted_before_first_attempt(
        self, db: Session, client: TestClient
    ):
        # Test that a run still retries if the pod is deleted before the first retry attempt starts.
        list_namespaced_pods_calls = [
            [self.pending_job_pod],
            [self.running_job_pod],
            # simulate deleted pod
            [],
            [self.failed_job_pod],
            # additional time for the get_logger_pods
            [self.failed_job_pod],
        ]
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_read_namespaced_pod_log()

        # Simulate that no runtime resources are found
        self.runtime_handler._get_runtime_resources = unittest.mock.Mock(
            return_value=[]
        )
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )

        # Store the run with retry spec
        self._store_run(
            db,
            retry_spec={"count": 3},
        )

        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)

        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )

        self._assert_run_reached_state(
            db,
            self.project,
            self.run_uid,
            RunStates.pending_retry,
            expected_status_attrs={
                "reason": "Some reason",
                "status_text": "Run failed attempt 1 of 4 with error: Failed message",
            },
        )

    @pytest.mark.parametrize(
        "pod_retry_label, run_retry_count, expected_result",
        [
            # first run, no retry label means pod is valid and not outdated
            (None, 0, False),
            # retry count > 0 and no retry label is present, pod is outdated
            (None, 1, True),
            # pod attempt is older than current run retry, pod is outdated
            ("1", 2, True),
            # pod attempt equals current run retry, pod is still valid
            ("2", 2, False),
            # edge case: pod attempt label is ahead of the run's retry count.
            # this situation shouldn't normally occur, but if it does (e.g. due to a transient state or race condition),
            # we treat the pod as valid (not outdated) to avoid skipping an active attempt.
            ("3", 2, False),
        ],
    )
    def test_is_pod_from_outdated_retry(
        self, pod_retry_label, run_retry_count, expected_result
    ):
        pod = self._generate_pod("pod", self.job_labels, PodPhases.pending)
        if pod_retry_label is not None:
            pod.metadata.labels[mlrun.common.constants.MLRunInternalLabels.retry] = (
                pod_retry_label
            )
        self.run["status"]["retry_count"] = run_retry_count
        assert (
            self.runtime_handler._is_pod_from_outdated_retry(pod.to_dict(), self.run)
            is expected_result
        )

    def _mock_list_resources_pods(self, pod=None):
        pod = pod or self.completed_job_pod
        mocked_responses = self._mock_list_namespaced_pods([[pod]])
        return mocked_responses[0].items
