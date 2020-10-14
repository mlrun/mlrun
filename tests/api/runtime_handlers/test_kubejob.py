import unittest.mock
from datetime import datetime

from fastapi.testclient import TestClient
from kubernetes import client
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import RunStates, PodPhases
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.job)

        # initializing them here to save space in tests
        self.pending_pod = self._generate_pod(
            self.project, self.run_uid, PodPhases.pending,
        )
        self.running_pod = self._generate_pod(
            self.project, self.run_uid, PodPhases.running,
        )
        self.completed_pod = self._generate_pod(
            self.project, self.run_uid, PodPhases.succeeded,
        )
        self.failed_pod = self._generate_pod(
            self.project, self.run_uid, PodPhases.failed,
        )

    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, expected_pods=pods
        )

    def test_monitor_run_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_pod],
            [self.running_pod],
            [self.completed_pod],
            # additional time for the get_logger_pods
            [self.completed_pod],
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
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
        self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_pod.metadata.name,
        )

    def test_monitor_run_failed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_pod],
            [self.running_pod],
            [self.failed_pod],
            # additional time for the get_logger_pods
            [self.failed_pod],
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.failed_pod.metadata.name,
        )

    def test_monitor_run_no_pods(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [],
            [],
            [],
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.created
        )
        self._assert_run_logs(db, self.project, self.run_uid, "")

    def test_monitor_run_overriding_terminal_state(
        self, db: Session, client: TestClient
    ):
        list_namespaced_pods_calls = [
            [self.failed_pod],
            # additional time for the get_logger_pods
            [self.failed_pod],
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
        self.run["status"]["state"] = RunStates.completed
        get_db().store_run(db, self.run, self.run_uid, self.project)
        expected_monitor_cycles_to_reach_expected_state = (
            expected_number_of_list_pods_calls - 1
        )
        for _ in range(expected_monitor_cycles_to_reach_expected_state):
            self.runtime_handler.monitor_runs(get_db(), db)
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, expected_number_of_list_pods_calls
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_pod.metadata.name,
        )

    def test_monitor_run_run_does_not_exists(self, db: Session, client: TestClient):
        get_db().del_run(db, self.run_uid, self.project)
        list_namespaced_pods_calls = [
            [self.completed_pod],
            # additional time for the get_logger_pods
            [self.completed_pod],
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
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
        self._assert_run_logs(
            db,
            self.project,
            self.run_uid,
            log,
            self.completed_pod.metadata.name,
        )

    def _mock_list_resources_pods(self):
        mocked_responses = self._mock_list_namespaces_pods([[self.completed_pod]])
        return mocked_responses[0].items

    @staticmethod
    def _generate_pod(project, uid, phase=PodPhases.succeeded):
        terminated_container_state = client.V1ContainerStateTerminated(finished_at=datetime.now(), exit_code=0)
        container_state = client.V1ContainerState(terminated=terminated_container_state)
        container_status = client.V1ContainerStatus(state=container_state, image="must/provide:image", image_id="must-provide-image-id", name="must-provide-name", ready=True, restart_count=0)
        status = client.V1PodStatus(phase=phase, container_statuses=[container_status])

        labels = {
            "mlrun/class": "job",
            "mlrun/function": "my-trainer",
            "mlrun/name": "my-training",
            "mlrun/project": project,
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "mlrun/uid": uid,
        }
        metadata = client.V1ObjectMeta(name="my-training-j7dtf", labels=labels)
        pod = client.V1Pod(metadata=metadata, status=status)
        return pod
