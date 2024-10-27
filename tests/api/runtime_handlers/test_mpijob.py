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
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from kubernetes import client as k8s_client
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import server.api.runtime_handlers.mpijob
import server.api.utils.helpers
from mlrun.common.runtimes.constants import PodPhases, RunStates
from mlrun.runtimes import RuntimeKinds
from server.api.runtime_handlers import get_runtime_handler
from server.api.utils.singletons.db import get_db
from server.api.utils.singletons.k8s import get_k8s_helper
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestMPIjobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.kind = RuntimeKinds.mpijob
        self.runtime_handler = get_runtime_handler(RuntimeKinds.mpijob)
        self.runtime_handler.wait_for_deletion_interval = 0

        # initializing them here to save space in tests
        self.active_crd_dict = self._generate_mpijob_crd(
            self.project,
            self.run_uid,
            self._get_active_crd_status(),
        )
        self.succeeded_crd_dict = self._generate_mpijob_crd(
            self.project,
            self.run_uid,
            self._get_succeeded_crd_status(),
        )
        self.failed_crd_dict = self._generate_mpijob_crd(
            self.project,
            self.run_uid,
            self._get_failed_crd_status(),
        )
        self.no_status_crd_dict = self._generate_mpijob_crd(
            self.project,
            self.run_uid,
        )

        self.launcher_pod_labels = {
            "group-name": "kubeflow.org",
            mlrun_constants.MLRunInternalLabels.mlrun_class: "mpijob",
            mlrun_constants.MLRunInternalLabels.function: "trainer",
            mlrun_constants.MLRunInternalLabels.job: "trainer-1b019005",
            mlrun_constants.MLRunInternalLabels.name: "trainer",
            mlrun_constants.MLRunInternalLabels.mlrun_owner: "iguazio",
            mlrun_constants.MLRunInternalLabels.project: self.project,
            mlrun_constants.MLRunInternalLabels.scrape_metrics: "True",
            mlrun_constants.MLRunInternalLabels.tag: "latest",
            mlrun_constants.MLRunInternalLabels.uid: self.run_uid,
            "mpi-job-name": "trainer-1b019005",
            mlrun_constants.MLRunInternalLabels.mpi_job_role: "launcher",
        }
        launcher_pod_name = "trainer-1b019005-launcher"

        self.launcher_pod = self._generate_pod(
            launcher_pod_name,
            self.launcher_pod_labels,
            PodPhases.running,
        )

        self.worker_pod_labels = {
            "group-name": "kubeflow.org",
            mlrun_constants.MLRunInternalLabels.mlrun_class: "mpijob",
            mlrun_constants.MLRunInternalLabels.function: "trainer",
            mlrun_constants.MLRunInternalLabels.job: "trainer-1b019005",
            mlrun_constants.MLRunInternalLabels.name: "trainer",
            mlrun_constants.MLRunInternalLabels.mlrun_owner: "iguazio",
            mlrun_constants.MLRunInternalLabels.project: self.project,
            mlrun_constants.MLRunInternalLabels.schedule_name: "True",
            mlrun_constants.MLRunInternalLabels.tag: "latest",
            mlrun_constants.MLRunInternalLabels.uid: self.run_uid,
            "mpi-job-name": "trainer-1b019005",
            mlrun_constants.MLRunInternalLabels.mpi_job_role: "worker",
        }
        worker_pod_name = "trainer-1b019005-worker-0"

        self.worker_pod = self._generate_pod(
            worker_pod_name,
            self.worker_pod_labels,
            PodPhases.running,
        )

        self.pod_label_selector = self._generate_get_logger_pods_label_selector(
            self.runtime_handler
        )

    def test_list_resources(self, db: Session, client: TestClient):
        mocked_responses = self._mock_list_namespaced_crds([[self.succeeded_crd_dict]])
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.mpijob,
            expected_crds=mocked_responses[0]["items"],
            expected_pods=pods,
        )

    def test_list_resources_with_crds_without_status(
        self, db: Session, client: TestClient
    ):
        mocked_responses = self._mock_list_namespaced_crds([[self.no_status_crd_dict]])
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.mpijob,
            expected_crds=mocked_responses[0]["items"],
            expected_pods=pods,
        )

    def test_list_resources_grouped_by_job(self, db: Session, client: TestClient):
        for group_by in [
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.job,
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.project,
        ]:
            mocked_responses = self._mock_list_namespaced_crds(
                [[self.succeeded_crd_dict]]
            )
            pods = self._mock_list_resources_pods()
            self._assert_runtime_handler_list_resources(
                RuntimeKinds.mpijob,
                expected_crds=mocked_responses[0]["items"],
                expected_pods=pods,
                group_by=group_by,
            )

    @pytest.mark.asyncio
    async def test_delete_resources_succeeded_crd(
        self, db: Session, client: TestClient
    ):
        list_namespaced_crds_calls = [
            [self.succeeded_crd_dict],
            # 2 additional time for wait for pods deletion
            [self.succeeded_crd_dict],
            [self.succeeded_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        list_namespaced_pods_calls = [
            # for the get_logger_pods
            [self.launcher_pod, self.worker_pod],
            # additional time for wait for pods deletion - simulate pods not removed yet
            [self.launcher_pod, self.worker_pod],
            # additional time for wait for pods deletion - simulate pods gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_custom_objects()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.succeeded_crd_dict["metadata"]["name"]],
            self.succeeded_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls), paginated=False
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
            paginated=False,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.launcher_pod.metadata.name,
        )

    def test_delete_resources_running_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)

        # nothing removed cause crd is active
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls), paginated=False
        )

    def test_delete_resources_with_grace_period(self, db: Session, client: TestClient):
        recently_completed_crd_dict = self._generate_mpijob_crd(
            self.project,
            self.run_uid,
            self._get_succeeded_crd_status(datetime.now(timezone.utc).isoformat()),
        )
        list_namespaced_crds_calls = [
            [recently_completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10)

        # nothing removed cause grace period didn't pass
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls), paginated=False
        )

    @pytest.mark.asyncio
    async def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
            # additional time for wait for pods deletion
            [self.active_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        list_namespaced_pods_calls = [
            # for the get_logger_pods
            [self.launcher_pod, self.worker_pod],
            # additional time for wait for pods deletion - simulate pods gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_custom_objects()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10, force=True)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.active_crd_dict["metadata"]["name"]],
            self.active_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls), paginated=False
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
            paginated=False,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.launcher_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_succeeded_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
            [self.succeeded_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            # 1 call per threshold state verification or for logs collection (runs in terminal state)
            [],
            [self.launcher_pod, self.worker_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_crds_calls
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            # 1 call per threshold state verification or for logs collection (runs in terminal state)
            2,
            self.pod_label_selector,
            paginated=False,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.launcher_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_failed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.active_crd_dict],
            [self.failed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            # 1 call per threshold state verification or for logs collection (runs in terminal state)
            [],
            [self.launcher_pod, self.worker_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        log = self._mock_read_namespaced_pod_log()
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_crds_calls
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
            paginated=False,
        )
        self._assert_run_reached_state(
            db,
            self.project,
            self.run_uid,
            RunStates.error,
            expected_status_attrs={
                "reason": "Some reason",
                "status_text": "Some message",
            },
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.launcher_pod.metadata.name,
        )

    def test_state_thresholds(self, db: Session, client: TestClient):
        """
        Test that the runtime handler aborts runs that are in a state for too long
        Creates 4 CRDs:
        1. job in image pull backoff that should be aborted
        2. job in running state that should be aborted
        3. job in running state that should not be aborted
        4. job in succeeded state that should not be aborted
        """
        # set big debouncing interval to avoid having to mock resources for all the runs on every monitor cycle
        mlrun.mlconf.monitoring.runs.missing_runtime_resources_debouncing_interval = (
            server.api.utils.helpers.time_string_to_seconds(
                mlrun.mlconf.function.spec.state_thresholds.default.executing
            )
            * 2
        )
        image_pull_backoff_job_uid = str(uuid.uuid4())
        running_long_uid = str(uuid.uuid4())
        running_short_uid = str(uuid.uuid4())
        success_uid = str(uuid.uuid4())

        # create the runs
        for uid, start_time in [
            (
                image_pull_backoff_job_uid,
                datetime.now(timezone.utc)
                - timedelta(
                    seconds=server.api.utils.helpers.time_string_to_seconds(
                        mlrun.mlconf.function.spec.state_thresholds.default.image_pull_backoff
                    )
                ),
            ),
            (
                running_long_uid,
                datetime.now(timezone.utc)
                - timedelta(
                    seconds=server.api.utils.helpers.time_string_to_seconds(
                        mlrun.mlconf.function.spec.state_thresholds.default.executing
                    )
                ),
            ),
            (running_short_uid, datetime.now(timezone.utc)),
            (success_uid, datetime.now(timezone.utc)),
        ]:
            self._store_run(
                db,
                "train",
                uid,
                start_time=start_time,
            )

        # create image pull backoff job
        image_pull_backoff_job = self._generate_mpijob_crd(
            self.project,
            image_pull_backoff_job_uid,
            self._get_active_crd_status(),
        )

        # create running long job
        running_long_job = self._generate_mpijob_crd(
            self.project,
            running_long_uid,
            self._get_active_crd_status(),
        )

        # create running short job
        running_short_job = self._generate_mpijob_crd(
            self.project,
            running_short_uid,
            self._get_active_crd_status(),
        )

        list_namespaced_crds_calls = [
            [image_pull_backoff_job, running_short_job],
            [running_long_job, self.succeeded_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)

        # create image pull backoff pods
        worker_pod_image_pull_backoff = self._generate_pod(
            "worker_in_image_pull_backoff",
            self._generate_job_labels(
                image_pull_backoff_job,
                uid=image_pull_backoff_job_uid,
                job_labels=self.worker_pod_labels,
            ),
            PodPhases.pending,
        )
        worker_pod_image_pull_backoff.status.container_statuses = [
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
        launcher_pod_image_pull_backoff = self._generate_pod(
            "launcher",
            self._generate_job_labels(
                image_pull_backoff_job["metadata"]["labels"][
                    mlrun_constants.MLRunInternalLabels.name
                ],
                uid=image_pull_backoff_job_uid,
                job_labels=self.launcher_pod_labels,
            ),
            PodPhases.running,
        )
        list_namespaced_pods_calls = [
            [launcher_pod_image_pull_backoff, worker_pod_image_pull_backoff],
        ]

        # create running pods
        for name, uid, job in [
            ("running-long", running_long_uid, running_long_job),
            ("running-short", running_short_uid, running_short_job),
        ]:
            worker_pod = self._generate_pod(
                f"worker-{name}",
                self._generate_job_labels(
                    job["metadata"]["labels"][mlrun_constants.MLRunInternalLabels.name],
                    uid=uid,
                    job_labels=self.worker_pod_labels,
                ),
                PodPhases.running,
            )
            launcher_pod = self._generate_pod(
                f"launcher-{name}",
                self._generate_job_labels(
                    job["metadata"]["labels"][mlrun_constants.MLRunInternalLabels.name],
                    uid=uid,
                    job_labels=self.launcher_pod_labels,
                ),
                PodPhases.running,
            )
            list_namespaced_pods_calls.append([launcher_pod, worker_pod])

        # mock succeeded pods
        list_namespaced_pods_calls.append([])
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_read_namespaced_pod_log()
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_crds_calls
        )

        stale_runs = []
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            stale_runs.extend(self.runtime_handler.monitor_runs(get_db(), db))

        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            # 1 call per threshold state verification or for logs collection (runs in terminal state)
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
            paginated=False,
        )
        assert len(stale_runs) == 2

        stale_run_uids = [run["uid"] for run in stale_runs]
        expected_stale_run_uids = [
            image_pull_backoff_job_uid,
            running_long_uid,
        ]
        assert stale_run_uids == expected_stale_run_uids

        stale_run_updates = [run["run_updates"] for run in stale_runs]
        expected_run_updates = []
        for state in ["image_pull_backoff", "executing"]:
            expected_run_updates.append(
                {
                    "status.error": f"Run aborted due to exceeded state threshold: {state}",
                }
            )
        assert stale_run_updates == expected_run_updates

    def test_state_thresholds_pending_states(self, db: Session, client: TestClient):
        # set big debouncing interval to avoid having to mock resources for all the runs on every monitor cycle
        mlrun.mlconf.monitoring.runs.missing_runtime_resources_debouncing_interval = (
            server.api.utils.helpers.time_string_to_seconds(
                mlrun.mlconf.function.spec.state_thresholds.default.pending_scheduled
            )
            * 2
        )
        pending_uid = str(uuid.uuid4())
        pending_scheduled_stale_uid = str(uuid.uuid4())
        pending_scheduled_uid = str(uuid.uuid4())

        # create the runs
        for name, uid, start_time in [
            ("pending", pending_uid, datetime.now(timezone.utc)),
            (
                "pending-scheduled-stale",
                pending_scheduled_stale_uid,
                datetime.now(timezone.utc)
                - timedelta(
                    seconds=server.api.utils.helpers.time_string_to_seconds(
                        mlrun.mlconf.function.spec.state_thresholds.default.pending_scheduled
                    )
                ),
            ),
            (
                "pending-scheduled",
                pending_scheduled_uid,
                datetime.now(timezone.utc),
            ),
        ]:
            self._store_run(
                db,
                f"train={name}",
                uid,
                start_time=start_time,
            )

        # create crds job
        list_namespaced_crds_calls = [[]]
        for uid in [pending_scheduled_stale_uid, pending_scheduled_uid, pending_uid]:
            list_namespaced_crds_calls[0].append(
                self._generate_mpijob_crd(
                    self.project,
                    uid,
                    self._get_active_crd_status(),
                )
            )
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)

        # create scheduled pods
        list_namespaced_pods_calls = []
        for name, uid in [
            ("pending-scheduled-stale", pending_scheduled_stale_uid),
            ("pending-scheduled", pending_scheduled_uid),
            ("pending", pending_uid),
        ]:
            worker_pod = self._generate_pod(
                f"worker-{name}",
                self._generate_job_labels(
                    name,
                    uid=uid,
                    job_labels=self.worker_pod_labels,
                ),
                PodPhases.pending,
            )
            launcher_pod = self._generate_pod(
                f"launcher-{name}",
                self._generate_job_labels(
                    name,
                    uid=uid,
                    job_labels=self.launcher_pod_labels,
                ),
                PodPhases.pending,
            )
            if "scheduled" in name:
                worker_pod.status.conditions = [
                    k8s_client.V1PodCondition(type="PodScheduled", status="True")
                ]
                launcher_pod.status.conditions = [
                    k8s_client.V1PodCondition(type="PodScheduled", status="True")
                ]

            list_namespaced_pods_calls.append([launcher_pod, worker_pod])

        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)

        stale_runs = self.runtime_handler.monitor_runs(get_db(), db)

        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            # 1 call per threshold state verification
            len(list_namespaced_pods_calls),
            f"{mlrun_constants.MLRunInternalLabels.uid}={pending_scheduled_stale_uid}",
            paginated=False,
        )
        assert len(stale_runs) == 1

        stale_run_uids = [run["uid"] for run in stale_runs]
        expected_stale_run_uids = [
            pending_scheduled_stale_uid,
        ]
        assert stale_run_uids == expected_stale_run_uids

        stale_run_updates = [run["run_updates"] for run in stale_runs]
        expected_run_updates = []
        for state in ["pending_scheduled"]:
            expected_run_updates.append(
                {
                    "status.error": f"Run aborted due to exceeded state threshold: {state}",
                }
            )
        assert stale_run_updates == expected_run_updates

    def _mock_list_resources_pods(self):
        mocked_responses = self._mock_list_namespaced_pods(
            [[self.launcher_pod, self.worker_pod]]
        )
        return mocked_responses[0].items

    def _generate_get_logger_pods_label_selector(self, runtime_handler):
        logger_pods_label_selector = super()._generate_get_logger_pods_label_selector(
            runtime_handler
        )
        return f"{logger_pods_label_selector},{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher"

    @staticmethod
    def _generate_mpijob_crd(project, uid, status=None):
        crd_dict = {
            "metadata": {
                "name": "train-eaf63df8",
                "namespace": get_k8s_helper().resolve_namespace(),
                "labels": {
                    mlrun_constants.MLRunInternalLabels.mlrun_class: "mpijob",
                    mlrun_constants.MLRunInternalLabels.function: "trainer",
                    mlrun_constants.MLRunInternalLabels.name: "train",
                    mlrun_constants.MLRunInternalLabels.project: project,
                    mlrun_constants.MLRunInternalLabels.scrape_metrics: "False",
                    mlrun_constants.MLRunInternalLabels.tag: "latest",
                    mlrun_constants.MLRunInternalLabels.uid: uid,
                },
            },
        }
        if status is not None:
            crd_dict["status"] = status
        return crd_dict

    @staticmethod
    def _get_active_crd_status(start_timestamp=None):
        return {
            "startTime": start_timestamp or "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"active": 1}, "Worker": {"active": 4}},
        }

    @staticmethod
    def _get_succeeded_crd_status(completion_timestamp=None, start_timestamp=None):
        return {
            "startTime": start_timestamp or "2020-10-06T00:36:41Z",
            "completionTime": completion_timestamp or "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"succeeded": 1}, "Worker": {}},
        }

    @staticmethod
    def _get_failed_crd_status():
        return {
            "completionTime": "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"failed": 1}, "Worker": {}},
            "conditions": [{"reason": "Some reason", "message": "Some message"}],
        }
