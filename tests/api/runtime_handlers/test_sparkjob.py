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
import copy
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from kubernetes import client as k8s_client
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import server.api.utils.helpers
from mlrun.common.runtimes.constants import PodPhases, RunStates
from mlrun.runtimes import RuntimeKinds
from server.api.runtime_handlers import get_runtime_handler
from server.api.utils.singletons.db import get_db
from server.api.utils.singletons.k8s import get_k8s_helper
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestSparkjobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.kind = RuntimeKinds.spark
        self.runtime_handler = get_runtime_handler(RuntimeKinds.spark)
        self.runtime_handler.wait_for_deletion_interval = 0
        self._ui_url = "http://spark-ui-url:4040"

        # initializing them here to save space in tests
        self.running_crd_dict = self._generate_sparkjob_crd(
            self.project,
            self.run_uid,
            self._get_running_crd_status(driver_ui_url=self._ui_url),
        )
        self.completed_crd_dict = self._generate_sparkjob_crd(
            self.project,
            self.run_uid,
            self._get_completed_crd_status(driver_ui_url=self._ui_url),
        )
        self.failed_crd_dict = self._generate_sparkjob_crd(
            self.project,
            self.run_uid,
            self._get_failed_crd_status(driver_ui_url=self._ui_url),
        )

        executor_pod_labels = {
            mlrun_constants.MLRunInternalLabels.mlrun_class: "spark",
            mlrun_constants.MLRunInternalLabels.function: "my-spark-jdbc",
            mlrun_constants.MLRunInternalLabels.job: "my-spark-jdbc-2ea432f1",
            mlrun_constants.MLRunInternalLabels.name: "my-spark-jdbc",
            mlrun_constants.MLRunInternalLabels.project: self.project,
            mlrun_constants.MLRunInternalLabels.uid: self.run_uid,
            mlrun_constants.MLRunInternalLabels.scrape_metrics: "False",
            mlrun_constants.MLRunInternalLabels.tag: "latest",
            "spark-app-selector": "spark-12f88a73cb544ce298deba34947226a4",
            "spark-exec-id": "1",
            "spark-role": "executor",
            "sparkoperator.k8s.io/app-name": "my-spark-jdbc-2ea432f1",
            "sparkoperator.k8s.io/launched-by-spark-operator": "true",
            "sparkoperator.k8s.io/submission-id": "44343f6b-42ca-41d4-b01a-66052cc5c919",
        }
        executor_pod_name = "my-spark-jdbc-2ea432f1-1597760338437-exec-1"

        self.executor_pod = self._generate_pod(
            executor_pod_name,
            executor_pod_labels,
            PodPhases.running,
        )

        self.driver_pod_labels = {
            mlrun_constants.MLRunInternalLabels.mlrun_class: "spark",
            mlrun_constants.MLRunInternalLabels.function: "my-spark-jdbc",
            mlrun_constants.MLRunInternalLabels.job: "my-spark-jdbc-2ea432f1",
            mlrun_constants.MLRunInternalLabels.name: "my-spark-jdbc",
            mlrun_constants.MLRunInternalLabels.project: self.project,
            mlrun_constants.MLRunInternalLabels.uid: self.run_uid,
            mlrun_constants.MLRunInternalLabels.scrape_metrics: "False",
            mlrun_constants.MLRunInternalLabels.tag: "latest",
            "spark-app-selector": "spark-12f88a73cb544ce298deba34947226a4",
            "spark-role": "driver",
            "sparkoperator.k8s.io/app-name": "my-spark-jdbc-2ea432f1",
            "sparkoperator.k8s.io/launched-by-spark-operator": "true",
            "sparkoperator.k8s.io/submission-id": "44343f6b-42ca-41d4-b01a-66052cc5c919",
        }
        driver_pod_name = "my-spark-jdbc-2ea432f1-driver"

        self.driver_pod = self._generate_pod(
            driver_pod_name,
            self.driver_pod_labels,
            PodPhases.running,
        )

        self.pod_label_selector = self._generate_get_logger_pods_label_selector(
            self.runtime_handler
        )

        self.config_map = self._generate_config_map(
            name="my-spark-jdbc",
            labels={mlrun_constants.MLRunInternalLabels.uid: self.run_uid},
        )

    def test_list_resources(self, db: Session, client: TestClient):
        mocked_responses = self._mock_list_namespaced_crds([[self.completed_crd_dict]])
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.spark,
            expected_crds=mocked_responses[0]["items"],
            expected_pods=pods,
        )

    def test_list_resources_grouped_by_job(self, db: Session, client: TestClient):
        for group_by in [
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.job,
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.project,
        ]:
            mocked_responses = self._mock_list_namespaced_crds(
                [[self.completed_crd_dict]]
            )
            pods = self._mock_list_resources_pods()
            self._assert_runtime_handler_list_resources(
                RuntimeKinds.spark,
                expected_crds=mocked_responses[0]["items"],
                expected_pods=pods,
                group_by=group_by,
            )

    @pytest.mark.asyncio
    async def test_delete_resources_completed_crd(
        self, db: Session, client: TestClient
    ):
        list_namespaced_crds_calls = [
            [self.completed_crd_dict],
            # 2 additional time for wait for pods deletion
            [self.completed_crd_dict],
            [self.completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        list_namespaced_pods_calls = [
            # for the get_logger_pods with proper selector
            [self.driver_pod],
            # additional time for wait for pods deletion - simulate pods not removed yet
            [self.executor_pod, self.driver_pod],
            # additional time for wait for pods deletion - simulate pods gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_list_namespaced_config_map([self.config_map])
        self._mock_delete_namespaced_custom_objects()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.completed_crd_dict["metadata"]["name"]],
            self.completed_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            len(list_namespaced_crds_calls),
            paginated=False,
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
            self.driver_pod.metadata.name,
        )

    def test_delete_resources_running_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_list_namespaced_config_map([self.config_map])
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db)

        # nothing removed cause crd is running
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            len(list_namespaced_crds_calls),
            paginated=False,
        )

    def test_delete_resources_with_grace_period(self, db: Session, client: TestClient):
        recently_completed_crd_dict = self._generate_sparkjob_crd(
            self.project,
            self.run_uid,
            self._get_completed_crd_status(datetime.now(timezone.utc).isoformat()),
        )
        list_namespaced_crds_calls = [
            [recently_completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_list_namespaced_config_map([self.config_map])
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
            paginated=False,
        )

    @pytest.mark.asyncio
    async def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            # additional time for wait for pods deletion
            [self.completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        list_namespaced_pods_calls = [
            # for the get_logger_pods with proper selector
            [self.driver_pod],
            # additional time for wait for pods deletion - simulate pods gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_list_namespaced_config_map([self.config_map])
        self._mock_delete_namespaced_custom_objects()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, force=True)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.running_crd_dict["metadata"]["name"]],
            self.running_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            len(list_namespaced_crds_calls),
            paginated=False,
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
            self.driver_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_completed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            [self.completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods with proper selector
        list_namespaced_pods_calls = [
            # 1 call per threshold state verification or for logs collection (runs in terminal state)
            [],
            [self.driver_pod],
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
            db, self.project, self.run_uid, RunStates.completed
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.driver_pod.metadata.name,
        )

    @pytest.mark.asyncio
    async def test_monitor_run_failed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            [self.failed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods with proper selector
        list_namespaced_pods_calls = [
            # 1 call per threshold state verification or for logs collection (runs in terminal state)
            [],
            [self.driver_pod],
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
            expected_label_selector=self.pod_label_selector,
            paginated=False,
        )
        self._assert_run_reached_state(
            db,
            self.project,
            self.run_uid,
            RunStates.error,
            expected_status_attrs={"status_text": "Some message"},
        )
        await self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.driver_pod.metadata.name,
        )

    def test_monitor_run_update_ui_url(self, db: Session, client: TestClient):
        db_instance = get_db()
        db_instance.del_run(db, self.run_uid, self.project)

        list_namespaced_crds_calls = [
            [self.running_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self.runtime_handler.monitor_runs(db_instance, db)

        run = get_db().read_run(db, self.run_uid, self.project)
        assert run["status"]["state"] == RunStates.running
        assert run["status"]["ui_url"] == self._ui_url

    @pytest.mark.parametrize(
        "threshold_state",
        (
            "image_pull_backoff",
            "pending_scheduled",
            "executing",
        ),
    )
    def test_state_thresholds(self, db: Session, client: TestClient, threshold_state):
        """
        Creates 2 spark jobs with a pod in the given state.
        1 that exceeds the threshold and 1 that doesn't.
        Verifies that the one that exceeds the threshold is marked as stale.
        """
        stale_job_uid = str(uuid.uuid4())
        new_job_uid = str(uuid.uuid4())
        stale_run_name = "my-spark-stale"
        new_run_name = "my-spark-new"

        threshold_in_seconds = server.api.utils.helpers.time_string_to_seconds(
            getattr(
                mlrun.mlconf.function.spec.state_thresholds.default,
                threshold_state,
            )
        )
        # set big debouncing interval to avoid having to mock resources for all the runs on every monitor cycle
        mlrun.mlconf.monitoring.runs.missing_runtime_resources_debouncing_interval = (
            threshold_in_seconds * 2
        )

        # create the runs
        for uid, name, start_time in [
            (
                stale_job_uid,
                stale_run_name,
                datetime.now(timezone.utc) - timedelta(seconds=threshold_in_seconds),
            ),
            (new_job_uid, new_run_name, datetime.now(timezone.utc)),
        ]:
            self._store_run(
                db,
                name,
                uid,
                start_time=start_time,
            )

        # create the crd
        list_namespaced_crds_calls = [[]]
        for uid in [stale_job_uid, new_job_uid]:
            running_crd_dict = self._generate_sparkjob_crd(
                self.project,
                uid,
                self._get_running_crd_status(driver_ui_url=self._ui_url),
            )
            list_namespaced_crds_calls[0].append(running_crd_dict)

        self._mock_list_namespaced_crds(list_namespaced_crds_calls)

        # create the pods
        list_namespaced_pods_calls = []
        for uid, pod_name in [
            (stale_job_uid, stale_run_name),
            (new_job_uid, new_run_name),
        ]:
            pod_phase = (
                PodPhases.pending
                if threshold_state != "executing"
                else PodPhases.running
            )
            driver_pod = self._generate_pod(
                pod_name,
                self._generate_job_labels(
                    pod_name, uid, job_labels=self.driver_pod_labels
                ),
                pod_phase,
            )
            if pod_phase == PodPhases.pending:
                if threshold_state == "image_pull_backoff":
                    driver_pod.status.container_statuses = [
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
                elif threshold_state == "pending_scheduled":
                    driver_pod.status.conditions = [
                        k8s_client.V1PodCondition(
                            status="True",
                            type="PodScheduled",
                        )
                    ]
            list_namespaced_pods_calls.append([driver_pod])

        self._mock_list_namespaced_pods(list_namespaced_pods_calls)

        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        stale_runs = self.runtime_handler.monitor_runs(get_db(), db)

        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            expected_label_selector=f"{mlrun_constants.MLRunInternalLabels.uid}={stale_job_uid}",
            paginated=False,
        )

        assert len(stale_runs) == 1
        assert stale_job_uid in [run["uid"] for run in stale_runs]
        assert stale_runs[0]["run_updates"] == {
            "status.error": f"Run aborted due to exceeded state threshold: {threshold_state}",
        }

    @pytest.mark.parametrize(
        "force",
        (
            True,
            False,
        ),
    )
    def test_delete_resources_stateless_crd(
        self, db: Session, client: TestClient, force
    ):
        stateless_crd = copy.deepcopy(self.completed_crd_dict)
        stateless_crd["status"]["applicationState"]["state"] = None
        list_namespaced_crds_calls = [
            [stateless_crd],
        ]

        list_namespaced_pods_calls = []
        if force:
            # additional time for wait for pods deletion
            list_namespaced_crds_calls.append([self.completed_crd_dict])
            list_namespaced_pods_calls = [
                # for the get_logger_pods with proper selector
                [self.driver_pod],
                # additional time for wait for pods deletion - simulate pods gone
                [],
            ]
            self._mock_list_namespaced_pods(list_namespaced_pods_calls)

        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_list_namespaced_config_map([self.config_map])
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, force=force)

        # deletion was skipped
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [] if not force else [stateless_crd["metadata"]["name"]],
            [] if not force else stateless_crd["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            len(list_namespaced_crds_calls),
            paginated=False,
        )

        if force:
            self._assert_list_namespaced_pods_calls(
                self.runtime_handler,
                len(list_namespaced_pods_calls),
                self.pod_label_selector,
                paginated=False,
            )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.created
        )

    def _generate_get_logger_pods_label_selector(self, runtime_handler):
        logger_pods_label_selector = super()._generate_get_logger_pods_label_selector(
            runtime_handler
        )
        return f"{logger_pods_label_selector},spark-role=driver"

    def _mock_list_resources_pods(self):
        mocked_responses = self._mock_list_namespaced_pods(
            [[self.executor_pod, self.driver_pod]]
        )
        return mocked_responses[0].items

    @staticmethod
    def _generate_sparkjob_crd(project, uid, status=None):
        if status is None:
            status = TestSparkjobRuntimeHandler._get_completed_crd_status()
        crd_dict = {
            "metadata": {
                "name": "my-spark-jdbc-2ea432f1",
                "namespace": get_k8s_helper().resolve_namespace(),
                "labels": {
                    mlrun_constants.MLRunInternalLabels.mlrun_class: "spark",
                    mlrun_constants.MLRunInternalLabels.function: "my-spark-jdbc",
                    mlrun_constants.MLRunInternalLabels.name: "my-spark-jdbc",
                    mlrun_constants.MLRunInternalLabels.project: project,
                    mlrun_constants.MLRunInternalLabels.scrape_metrics: "False",
                    mlrun_constants.MLRunInternalLabels.tag: "latest",
                    mlrun_constants.MLRunInternalLabels.uid: uid,
                },
            },
            "status": status,
        }
        return crd_dict

    @staticmethod
    def _get_running_crd_status(driver_ui_url=None):
        return {
            "applicationState": {"state": "RUNNING"},
            "driverInfo": {
                "webUIIngressAddress": driver_ui_url,
            },
        }

    @staticmethod
    def _get_completed_crd_status(timestamp=None, driver_ui_url=None):
        return {
            "terminationTime": timestamp or "2020-10-05T21:17:11Z",
            "applicationState": {"state": "COMPLETED"},
            "driverInfo": {
                "webUIIngressAddress": driver_ui_url,
            },
        }

    @staticmethod
    def _get_failed_crd_status(driver_ui_url=None):
        return {
            "terminationTime": "2020-10-05T21:17:11Z",
            "applicationState": {"state": "FAILED", "errorMessage": "Some message"},
            "driverInfo": {
                "webUIIngressAddress": driver_ui_url,
            },
        }
