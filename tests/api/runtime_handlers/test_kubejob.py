import unittest.mock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud as crud
from mlrun.api.constants import LogSources
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import RunStates
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, expected_pods=pods
        )

    def test_monitor_run(self, db: Session, client: TestClient):
        project = "test_project"
        uid = "test_run_uid"
        expected_number_of_list_pods_calls = self._mock_monitor_run_pods()
        log = "Some log string"
        self._mock_monitor_run_logs(log)
        run = {"status": {"state": RunStates.created}}
        get_db().store_run(db, run, uid, project)
        runtime_handler = get_runtime_handler(RuntimeKinds.job)
        expected_label_selector = runtime_handler._get_run_label_selector(project, uid)
        runtime_handler.monitor_run(get_db(), db, project, uid, interval=0)
        TestKubejobRuntimeHandler._assert_list_namespaces_pods_calls(
            expected_number_of_list_pods_calls, expected_label_selector
        )
        TestKubejobRuntimeHandler._assert_run_reached_state(db, project, uid, RunStates.completed)
        TestKubejobRuntimeHandler._assert_run_logs(db, project, uid, log)

    @staticmethod
    def _assert_run_logs(db: Session, project: str, uid: str, expected_log: str):
        _, log = crud.Logs.get_log(
            db, project, uid, source=LogSources.PERSISTENCY
        )
        assert log == expected_log.encode()

    @staticmethod
    def _assert_run_reached_state(db: Session, project: str, uid: str, expected_state: str):
        run = get_db().read_run(db, uid, project)
        assert run['status']['state'] == expected_state

    @staticmethod
    def _assert_list_namespaces_pods_calls(expected_number_of_calls: int, expected_label_selector: str):
        assert get_k8s().v1api.list_namespaced_pod.call_count == expected_number_of_calls
        get_k8s().v1api.list_namespaced_pod.assert_any_call(
            get_k8s().resolve_namespace(), label_selector=expected_label_selector
        )

    @staticmethod
    def _mock_list_resources_pods():
        pod_dict = TestKubejobRuntimeHandler._generate_pod_dict()
        mocked_responses = TestKubejobRuntimeHandler._mock_list_namespaces_pods(
            [[pod_dict]]
        )
        return mocked_responses[0]

    @staticmethod
    def _mock_monitor_run_logs(log):
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)

    @staticmethod
    def _mock_monitor_run_pods():
        pending_pod_dict = TestKubejobRuntimeHandler._generate_pod_dict(
            TestKubejobRuntimeHandler._get_pending_pod_status()
        )
        running_pod_dict = TestKubejobRuntimeHandler._generate_pod_dict(
            TestKubejobRuntimeHandler._get_running_pod_status()
        )
        completed_pod_dict = TestKubejobRuntimeHandler._generate_pod_dict(
            TestKubejobRuntimeHandler._get_completed_pod_status()
        )

        calls = [
            [pending_pod_dict],
            [running_pod_dict],
            [completed_pod_dict],
            # additional time for the get_logger_pods
            [completed_pod_dict],
        ]

        TestKubejobRuntimeHandler._mock_list_namespaces_pods(calls)

        return len(calls)

    @staticmethod
    def _generate_pod_dict(status=None):
        if status is None:
            status = TestKubejobRuntimeHandler._get_completed_pod_status()
        pod_dict = {
            "metadata": {
                "name": "my-training-j7dtf",
                "labels": {
                    "mlrun/class": "job",
                    "mlrun/function": "my-trainer",
                    "mlrun/name": "my-training",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": "bba96b8313b640cd9143d7513000c47c",
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
