from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.runtimes.constants import PodPhases, RunStates
from mlrun.utils import now_date
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.job)

        labels = {
            "mlrun/class": self._get_class_name(),
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

    def _get_class_name(self):
        return "job"

    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, expected_pods=pods
        )

    def test_delete_resources_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.completed_pod],
            # additional time for the get_logger_pods
            [self.completed_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods(
            [self.completed_pod.metadata.name], self.completed_pod.metadata.namespace
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_pod.metadata.name,
        )

    def test_delete_resources_running_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_pod],
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
            [self.completed_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10)

        # nothing removed cause pod grace period didn't pass
        self._assert_delete_namespaced_pods([])
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_pod],
            # additional time for the get_logger_pods
            [self.running_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_pods()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10, force=True)
        self._assert_delete_namespaced_pods(
            [self.running_pod.metadata.name], self.running_pod.metadata.namespace
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.running_pod.metadata.name,
        )

    def test_monitor_run_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_pod],
            [self.running_pod],
            [self.completed_pod],
            # additional time for the get_logger_pods
            [self.completed_pod],
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
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
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
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        log = self._mock_read_namespaced_pod_log()
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
        self._mock_list_namespaced_pods([[self.running_pod]])

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
        self._mock_list_namespaced_pods([[self.running_pod]])

        # Triggering monitor cycle
        self.runtime_handler.monitor_runs(get_db(), db)

        # verifying monitoring was not debounced
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )

        # Mocking pod that is in terminal state (extra one for the log collection)
        self._mock_list_namespaced_pods([[self.completed_pod], [self.completed_pod]])

        # Mocking read log calls
        log = self._mock_read_namespaced_pod_log()

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
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.completed_pod.metadata.name,
        )

    def _mock_list_resources_pods(self):
        mocked_responses = self._mock_list_namespaced_pods([[self.completed_pod]])
        return mocked_responses[0].items
