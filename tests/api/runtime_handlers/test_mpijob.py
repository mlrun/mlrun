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
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.runtimes.constants import PodPhases, RunStates
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

        launcher_pod_labels = {
            "group-name": "kubeflow.org",
            "mlrun/class": "mpijob",
            "mlrun/function": "trainer",
            "mlrun/job": "trainer-1b019005",
            "mlrun/name": "trainer",
            "mlrun/owner": "iguazio",
            "mlrun/project": self.project,
            "mlrun/scrape-metrics": "True",
            "mlrun/tag": "latest",
            "mlrun/uid": self.run_uid,
            "mpi-job-name": "trainer-1b019005",
            "mpi-job-role": "launcher",
        }
        launcher_pod_name = "trainer-1b019005-launcher"

        self.launcher_pod = self._generate_pod(
            launcher_pod_name,
            launcher_pod_labels,
            PodPhases.running,
        )

        worker_pod_labels = {
            "group-name": "kubeflow.org",
            "mlrun/class": "mpijob",
            "mlrun/function": "trainer",
            "mlrun/job": "trainer-1b019005",
            "mlrun/name": "trainer",
            "mlrun/owner": "iguazio",
            "mlrun/project": self.project,
            "mlrun/scrape-metrics": "True",
            "mlrun/tag": "latest",
            "mlrun/uid": self.run_uid,
            "mpi-job-name": "trainer-1b019005",
            "mpi-job-role": "worker",
        }
        worker_pod_name = "trainer-1b019005-worker-0"

        self.worker_pod = self._generate_pod(
            worker_pod_name,
            worker_pod_labels,
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
            mlrun.api.schemas.ListRuntimeResourcesGroupByField.job,
            mlrun.api.schemas.ListRuntimeResourcesGroupByField.project,
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
    async def test_delete_resources_succeeded_crd(self, db: Session, client: TestClient):
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
            self.runtime_handler,
            len(list_namespaced_crds_calls),
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
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
            self.runtime_handler,
            len(list_namespaced_crds_calls),
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
            self.runtime_handler,
            len(list_namespaced_crds_calls),
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
            self.runtime_handler,
            len(list_namespaced_crds_calls),
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
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
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.launcher_pod.metadata.name,
        )

    def _mock_list_resources_pods(self):
        mocked_responses = self._mock_list_namespaced_pods(
            [[self.launcher_pod, self.worker_pod]]
        )
        return mocked_responses[0].items

    def _generate_get_logger_pods_label_selector(self, runtime_handler):
        logger_pods_label_selector = super()._generate_get_logger_pods_label_selector(
            runtime_handler
        )
        return f"{logger_pods_label_selector},mpi-job-role=launcher"

    @staticmethod
    def _generate_mpijob_crd(project, uid, status=None):
        crd_dict = {
            "metadata": {
                "name": "train-eaf63df8",
                "namespace": get_k8s().resolve_namespace(),
                "labels": {
                    "mlrun/class": "mpijob",
                    "mlrun/function": "trainer",
                    "mlrun/name": "train",
                    "mlrun/project": project,
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": uid,
                },
            },
        }
        if status is not None:
            crd_dict["status"] = status
        return crd_dict

    @staticmethod
    def _get_active_crd_status():
        return {
            "replicaStatuses": {"Launcher": {"active": 1}, "Worker": {"active": 4}},
        }

    @staticmethod
    def _get_succeeded_crd_status(timestamp=None):
        return {
            "completionTime": timestamp or "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"succeeded": 1}, "Worker": {}},
        }

    @staticmethod
    def _get_failed_crd_status():
        return {
            "completionTime": "2020-10-06T00:36:41Z",
            "replicaStatuses": {"Launcher": {"failed": 1}, "Worker": {}},
        }
