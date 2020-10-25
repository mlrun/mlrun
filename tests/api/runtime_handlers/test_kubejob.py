import unittest.mock
from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import RunStates, PodPhases
from mlrun.utils import now_date
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.job)

        labels = {
            "mlrun/class": "job",
            "mlrun/function": "my-trainer",
            "mlrun/name": "my-training",
            "mlrun/project": self.project,
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "mlrun/uid": self.run_uid,
        }
        pod_name = "my-training-j7dtf"

        # initializing them here to save space in tests
        self.pending_pod = self._generate_pod(pod_name, labels, PodPhases.pending)
        self.running_pod = self._generate_pod(pod_name, labels, PodPhases.running)
        self.completed_pod = self._generate_pod(pod_name, labels, PodPhases.succeeded)
        self.failed_pod = self._generate_pod(pod_name, labels, PodPhases.failed)

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
            db, self.project, self.run_uid, log, self.completed_pod.metadata.name,
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
            db, self.project, self.run_uid, log, self.failed_pod.metadata.name,
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
            db, self.project, self.run_uid, log, self.completed_pod.metadata.name,
        )

    def test_monitor_run_debouncing_non_terminal_state(
        self, db: Session, client: TestClient
    ):
        # set monitoring interval so debouncing will be active
        config.runs_monitoring_interval = 100

        # Mocking the SDK updating the Run's state to terminal state
        self.run["status"]["state"] = RunStates.completed
        self.run["status"]["last_update"] = now_date().isoformat()
        get_db().store_run(db, self.run, self.run_uid, self.project)

        # Mocking pod that is still in non-terminal state
        self._mock_list_namespaces_pods([[self.running_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

        # Mocking that update occurred before debounced period
        debounce_period = config.runs_monitoring_interval
        self.run["status"]["last_update"] = (
            now_date() - timedelta(seconds=float(2 * debounce_period))
        ).isoformat()
        get_db().store_run(db, self.run, self.run_uid, self.project)

        # Mocking pod that is still in non-terminal state
        self._mock_list_namespaces_pods([[self.running_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

        # Mocking pod that is in terminal state (extra one for the log collection)
        self._mock_list_namespaces_pods([[self.completed_pod], [self.completed_pod]])

        # Mocking read log calls
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )

        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_pod.metadata.name,
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
            db, self.project, self.run_uid, log, self.completed_pod.metadata.name,
        )

    def _mock_list_resources_pods(self):
        mocked_responses = self._mock_list_namespaces_pods([[self.completed_pod]])
        return mocked_responses[0].items
