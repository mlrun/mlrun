import unittest.mock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes import get_runtime_handler
from mlrun.runtimes.constants import RunStates
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestSparkjobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.spark)

        # initializing them here to save space in tests
        self.running_crd_dict = self._generate_sparkjob_crd(
            self.project, self.run_uid, self._get_running_crd_status(),
        )
        self.completed_crd_dict = self._generate_sparkjob_crd(
            self.project, self.run_uid, self._get_completed_crd_status(),
        )
        self.failed_crd_dict = self._generate_sparkjob_crd(
            self.project, self.run_uid, self._get_failed_crd_status(),
        )

    def test_list_sparkjob_resources(self):
        mocked_responses = self._mock_list_namespaced_crds([[self.completed_crd_dict]])
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.spark,
            expected_crds=mocked_responses[0]["items"],
            expected_pods=pods,
        )

    def test_monitor_run_completed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            [self.completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            list(self._generate_pod_dicts()),
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        expected_crd_label_selector = self.runtime_handler._get_run_label_selector(
            self.project, self.run_uid
        )
        expected_pod_label_selector = f"mlrun/class,{expected_crd_label_selector}"
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
        self.runtime_handler.monitor_run(
            get_db(), db, self.project, self.run_uid, interval=0
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
            expected_crd_label_selector,
        )
        self._assert_list_namespaced_pods_calls(
            expected_number_of_list_pods_calls, expected_pod_label_selector
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(db, self.project, self.run_uid, log)

    def test_monitor_run_failed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            [self.failed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            list(self._generate_pod_dicts()),
        ]
        self._mock_list_namespaces_pods(list_namespaced_pods_calls)
        expected_number_of_list_crds_calls = len(list_namespaced_crds_calls)
        expected_number_of_list_pods_calls = len(list_namespaced_pods_calls)
        expected_crd_label_selector = self.runtime_handler._get_run_label_selector(
            self.project, self.run_uid
        )
        expected_pod_label_selector = f"mlrun/class,{expected_crd_label_selector}"
        log = "Some log string"
        get_k8s().v1api.read_namespaced_pod_log = unittest.mock.Mock(return_value=log)
        self.runtime_handler.monitor_run(
            get_db(), db, self.project, self.run_uid, interval=0
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler,
            expected_number_of_list_crds_calls,
            expected_crd_label_selector,
        )
        self._assert_list_namespaced_pods_calls(
            expected_number_of_list_pods_calls, expected_pod_label_selector
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(db, self.project, self.run_uid, log)

    @staticmethod
    def _generate_sparkjob_crd(project, uid, status=None):
        if status is None:
            status = TestSparkjobRuntimeHandler._get_completed_crd_status()
        crd_dict = {
            "metadata": {
                "name": "my-spark-jdbc-2ea432f1",
                "labels": {
                    "mlrun/class": "spark",
                    "mlrun/function": "my-spark-jdbc",
                    "mlrun/name": "my-spark-jdbc",
                    "mlrun/project": project,
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": uid,
                },
            },
            "status": status,
        }
        return crd_dict

    @staticmethod
    def _get_running_crd_status():
        return {
            "applicationState": {"state": "RUNNING"},
        }

    @staticmethod
    def _get_completed_crd_status():
        return {
            "terminationTime": "2020-10-05T21:17:11Z",
            "applicationState": {"state": "COMPLETED"},
        }

    @staticmethod
    def _get_failed_crd_status():
        return {
            "terminationTime": "2020-10-05T21:17:11Z",
            "applicationState": {"state": "FAILED"},
        }

    @staticmethod
    def _mock_list_resources_pods():
        (
            executor_pod_dict,
            driver_pod_dict,
        ) = TestSparkjobRuntimeHandler._generate_pod_dicts()
        mocked_responses = TestSparkjobRuntimeHandler._mock_list_namespaces_pods(
            [[executor_pod_dict, driver_pod_dict]]
        )
        return mocked_responses[0].items

    @staticmethod
    def _generate_pod_dicts():
        executor_pod_dict = {
            "metadata": {
                "name": "my-spark-jdbc-2ea432f1-1597760338437-exec-1",
                "labels": {
                    "mlrun/class": "spark",
                    "mlrun/function": "my-spark-jdbc",
                    "mlrun/job": "my-spark-jdbc-2ea432f1",
                    "mlrun/name": "my-spark-jdbc",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": "b532ba206a1649da9925d340d6f97f7a",
                    "spark-app-selector": "spark-12f88a73cb544ce298deba34947226a4",
                    "spark-exec-id": "1",
                    "spark-role": "executor",
                    "sparkoperator.k8s.io/app-name": "my-spark-jdbc-2ea432f1",
                    "sparkoperator.k8s.io/launched-by-spark-operator": "true",
                    "sparkoperator.k8s.io/submission-id": "44343f6b-42ca-41d4-b01a-66052cc5c919",
                },
            },
            "status": {
                "container_statuses": [
                    {
                        "state": {
                            "running": {"started_at": "2020-08-18T14:19:28+00:00"},
                            "terminated": None,
                            "waiting": None,
                        },
                    }
                ],
                "phase": "Running",
            },
        }
        driver_pod_dict = {
            "metadata": {
                "name": "my-spark-jdbc-2ea432f1-driver",
                "labels": {
                    "mlrun/class": "spark",
                    "mlrun/function": "my-spark-jdbc",
                    "mlrun/job": "my-spark-jdbc-2ea432f1",
                    "mlrun/name": "my-spark-jdbc",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": "b532ba206a1649da9925d340d6f97f7a",
                    "spark-app-selector": "spark-12f88a73cb544ce298deba34947226a4",
                    "spark-role": "driver",
                    "sparkoperator.k8s.io/app-name": "my-spark-jdbc-2ea432f1",
                    "sparkoperator.k8s.io/launched-by-spark-operator": "true",
                    "sparkoperator.k8s.io/submission-id": "44343f6b-42ca-41d4-b01a-66052cc5c919",
                },
            },
            "status": {
                "container_statuses": [
                    {
                        "state": {
                            "running": {"started_at": "2020-08-18T14:19:16+00:00"},
                            "terminated": None,
                            "waiting": None,
                        },
                    }
                ],
                "phase": "Running",
            },
        }
        return executor_pod_dict, driver_pod_dict
