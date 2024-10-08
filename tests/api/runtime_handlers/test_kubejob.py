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
import typing
import unittest.mock
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from kubernetes import client as k8s_client
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import server.api.crud
import server.api.utils.helpers
import server.api.utils.runtimes
import tests.conftest
from mlrun.common.runtimes.constants import PodPhases, RunStates
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.utils import now_date
from server.api.runtime_handlers import get_runtime_handler
from server.api.utils.singletons.db import get_db
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


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

    def test_ensure_run_not_stuck_on_non_terminal_state(
        self, db: Session, client: TestClient
    ):
        for test_case in [
            # no monitoring interval and no debouncing interval which means if run found in non-terminal state
            # the monitoring will override to terminal status
            {
                "runs_monitoring_interval": 0,
                "debouncing_interval": None,
                "list_namespaced_pods_calls": [[], []],
                "interval_time_to_add_to_run_update_time": 0,
                "start_run_states": RunStates.non_terminal_states(),
                "expected_reached_state": RunStates.error,
                "monitor_cycles": 1,
            },
            # monitoring interval and debouncing interval are configured which means debouncing interval will
            # be the debounce period, run is still in the debounce period that's why expecting not to override state
            # to terminal state
            {
                "runs_monitoring_interval": 30,
                "debouncing_interval": 100,
                "list_namespaced_pods_calls": [[], [], []],
                "interval_time_to_add_to_run_update_time": -70,
                "start_run_states": RunStates.non_terminal_states(),
                "expected_reached_state": RunStates.non_terminal_states(),
            },
            # monitoring interval and debouncing interval are configured which means debouncing interval will
            # be the debounce period, run update time passed the debounce period that's why expecting to override state
            # to terminal state
            {
                "runs_monitoring_interval": 30,
                "debouncing_interval": 100,
                "list_namespaced_pods_calls": [[], [], [], []],
                "interval_time_to_add_to_run_update_time": -200,
                "start_run_states": RunStates.non_terminal_states(),
                "expected_reached_state": RunStates.error,
                "monitor_cycles": 3,
            },
            # monitoring interval configured and debouncing interval isn't configured which means
            # monitoring interval * 2 will be the debounce period.
            # run isn't in the debounce period that's why expecting to override state to terminal state
            {
                "runs_monitoring_interval": 30,
                "debouncing_interval": None,
                "list_namespaced_pods_calls": [[], [], [], []],
                "interval_time_to_add_to_run_update_time": -65,
                "start_run_states": RunStates.non_terminal_states(),
                "expected_reached_state": RunStates.error,
                "monitor_cycles": 3,
            },
            # monitoring interval configured and debouncing interval isn't configured which means
            # monitoring interval * 2 will be the debounce period.
            # run is in the debounce period that's why expecting not to override state to terminal state
            {
                "runs_monitoring_interval": 30,
                "debouncing_interval": None,
                "list_namespaced_pods_calls": [[], [], []],
                "interval_time_to_add_to_run_update_time": -35,
                "start_run_states": RunStates.non_terminal_states(),
                "expected_reached_state": RunStates.non_terminal_states(),
            },
        ]:
            self._logger.info("running test case", test_case=test_case)
            config.monitoring.runs.interval = test_case.get(
                "runs_monitoring_interval", 0
            )

            config.monitoring.runs.missing_runtime_resources_debouncing_interval = (
                test_case.get("debouncing_interval", None)
            )

            list_namespaced_pods_calls = test_case.get(
                "list_namespaced_pods_calls", [[]]
            )
            interval_time_to_add_to_run_update_time = test_case.get(
                "interval_time_to_add_to_run_update_time", 0
            )
            expected_reached_state: typing.Union[str, list] = test_case.get(
                "expected_reached_state", RunStates.running
            )
            start_run_states = test_case.get("start_run_states", [RunStates.running])
            monitor_cycles = test_case.get(
                "monitor_cycles", len(list_namespaced_pods_calls)
            )
            for idx in range(len(start_run_states)):
                self.run["status"]["state"] = start_run_states[idx]

                # using freeze enables us to set the now attribute when calling the sub-function
                # _update_run_updated_time without the need to call the function directly
                original_update_run_updated_time = (
                    server.api.utils.singletons.db.get_db()._update_run_updated_time
                )
                server.api.utils.singletons.db.get_db()._update_run_updated_time = (
                    tests.conftest.freeze(
                        original_update_run_updated_time,
                        now=now_date()
                        + timedelta(
                            seconds=interval_time_to_add_to_run_update_time,
                        ),
                    )
                )
                server.api.crud.Runs().store_run(
                    db, self.run, self.run_uid, project=self.project
                )
                server.api.utils.singletons.db.get_db()._update_run_updated_time = (
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
        # set monitoring interval so debouncing will be active
        config.monitoring.runs.interval = 100

        # Mocking the SDK updating the Run's state to terminal state
        self.run["status"]["state"] = RunStates.completed
        original_update_run_updated_time = (
            server.api.utils.singletons.db.get_db()._update_run_updated_time
        )
        server.api.utils.singletons.db.get_db()._update_run_updated_time = (
            tests.conftest.freeze(original_update_run_updated_time, now=now_date())
        )
        server.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        server.api.utils.singletons.db.get_db()._update_run_updated_time = (
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
        server.api.utils.singletons.db.get_db()._update_run_updated_time = (
            tests.conftest.freeze(
                original_update_run_updated_time,
                now=now_date() - timedelta(seconds=float(2 * debounce_period)),
            )
        )
        server.api.crud.Runs().store_run(
            db, self.run, self.run_uid, project=self.project
        )
        server.api.utils.singletons.db.get_db()._update_run_updated_time = (
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

        # Mocking pod that is in terminal state (extra one for the log collection)
        self._mock_list_namespaced_pods(
            [[self.completed_job_pod], [self.completed_job_pod]]
        )

        # Mocking read log calls
        log = self._mock_read_namespaced_pod_log()

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
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
            seconds=server.api.utils.helpers.time_string_to_seconds(
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
            seconds=server.api.utils.helpers.time_string_to_seconds(
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
            seconds=server.api.utils.helpers.time_string_to_seconds(
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
            "server.api.db.sqldb.db.SQLDB.read_run",
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
            "server.api.db.sqldb.db.SQLDB.read_run", unittest.mock.Mock()
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

        server.api.crud.Runs().store_run(
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

    def _mock_list_resources_pods(self, pod=None):
        pod = pod or self.completed_job_pod
        mocked_responses = self._mock_list_namespaced_pods([[pod]])
        return mocked_responses[0].items
