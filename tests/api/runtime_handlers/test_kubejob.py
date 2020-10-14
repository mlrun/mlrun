import unittest.mock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import RunStates
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.job)

        # initializing them here to save space in tests
        self.pending_pod_dict = self._generate_pod_dict(
            self.project, self.run_uid, self._get_pending_pod_status(),
        )
        self.running_pod_dict = self._generate_pod_dict(
            self.project, self.run_uid, self._get_running_pod_status(),
        )
        self.completed_pod_dict = self._generate_pod_dict(
            self.project, self.run_uid, self._get_completed_pod_status(),
        )
        self.failed_pod_dict = self._generate_pod_dict(
            self.project, self.run_uid, self._get_failed_pod_status(),
        )

    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, expected_pods=pods
        )

    def test_monitor_run_completed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_pod_dict],
            [self.running_pod_dict],
            [self.completed_pod_dict],
            # additional time for the get_logger_pods
            [self.completed_pod_dict],
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
            self.completed_pod_dict["metadata"]["name"],
        )

    def test_monitor_run_failed_pod(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.pending_pod_dict],
            [self.running_pod_dict],
            [self.failed_pod_dict],
            # additional time for the get_logger_pods
            [self.failed_pod_dict],
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
            self.failed_pod_dict["metadata"]["name"],
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
            [self.failed_pod_dict],
            # additional time for the get_logger_pods
            [self.failed_pod_dict],
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
            self.completed_pod_dict["metadata"]["name"],
        )

    def test_monitor_run_run_does_not_exists(self, db: Session, client: TestClient):
        get_db().del_run(db, self.run_uid, self.project)
        list_namespaced_pods_calls = [
            [self.completed_pod_dict],
            # additional time for the get_logger_pods
            [self.completed_pod_dict],
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
            self.completed_pod_dict["metadata"]["name"],
        )

    def _mock_list_resources_pods(self):
        pod_dict = self._generate_pod_dict(self.project, self.run_uid)
        mocked_responses = self._mock_list_namespaces_pods([[pod_dict]])
        return mocked_responses[0].items

    @staticmethod
    def _generate_pod_dict(project, uid, status=None):
        if status is None:
            status = TestKubejobRuntimeHandler._get_completed_pod_status()
        pod_dict = {
            "metadata": {
                "name": "my-training-j7dtf",
                "labels": {
                    "mlrun/class": "job",
                    "mlrun/function": "my-trainer",
                    "mlrun/name": "my-training",
                    "mlrun/project": project,
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": uid,
                },
            },
            "status": status,
        }
        return pod_dict

    @staticmethod
    def _get_pending_pod_status():
        return {
            "container_statuses": [
                {
                    "state": {
                        "running": None,
                        "terminated": None,
                        "waiting": {"message": None, "reason": "ContainerCreating"},
                    },
                }
            ],
            "phase": "Pending",
        }

    @staticmethod
    def _get_running_pod_status():
        return {
            "container_statuses": [
                {
                    "state": {
                        "running": {"started_at": "2020-10-6T3:0:51+00:00"},
                        "terminated": None,
                        "waiting": None,
                    },
                }
            ],
            "phase": "Running",
        }

    @staticmethod
    def _get_completed_pod_status():
        return {
            "container_statuses": [
                {
                    "state": {
                        "running": None,
                        "terminated": {
                            "container_id": "docker://94a90870a6432d3140da821b87ae91980d21af2c000988fcb8687640a5f29886",
                            "exit_code": 0,
                            "finished_at": "2020-10-6T3:1:8+00:00",
                            "message": None,
                            "reason": "Completed",
                            "signal": None,
                            "started_at": "2020-10-6T3:1:8+00:00",
                        },
                        "waiting": None,
                    },
                }
            ],
            "phase": "Succeeded",
        }

    @staticmethod
    def _get_failed_pod_status():
        return {
            "container_statuses": [
                {
                    "state": {
                        "running": None,
                        "terminated": {
                            "container_id": "docker://ec259b0c68d9bc981964859ecac3d2da107b38da4fa7ca3df3c3eedb61bfb47e",
                            "exit_code": 1,
                            "finished_at": "2020-10-6T2:59:52+00:00",
                            "message": None,
                            "reason": "Error",
                            "signal": None,
                            "started_at": "2020-10-6T2:59:35+00:00",
                        },
                        "waiting": None,
                    },
                }
            ],
            "phase": "Failed",
        }
