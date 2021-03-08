from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.runtimes.constants import PodPhases, RunStates
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

        executor_pod_labels = {
            "mlrun/class": "spark",
            "mlrun/function": "my-spark-jdbc",
            "mlrun/job": "my-spark-jdbc-2ea432f1",
            "mlrun/name": "my-spark-jdbc",
            "mlrun/project": self.project,
            "mlrun/uid": self.run_uid,
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "spark-app-selector": "spark-12f88a73cb544ce298deba34947226a4",
            "spark-exec-id": "1",
            "spark-role": "executor",
            "sparkoperator.k8s.io/app-name": "my-spark-jdbc-2ea432f1",
            "sparkoperator.k8s.io/launched-by-spark-operator": "true",
            "sparkoperator.k8s.io/submission-id": "44343f6b-42ca-41d4-b01a-66052cc5c919",
        }
        executor_pod_name = "my-spark-jdbc-2ea432f1-1597760338437-exec-1"

        self.executor_pod = self._generate_pod(
            executor_pod_name, executor_pod_labels, PodPhases.running,
        )

        driver_pod_labels = {
            "mlrun/class": "spark",
            "mlrun/function": "my-spark-jdbc",
            "mlrun/job": "my-spark-jdbc-2ea432f1",
            "mlrun/name": "my-spark-jdbc",
            "mlrun/project": self.project,
            "mlrun/uid": self.run_uid,
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "spark-app-selector": "spark-12f88a73cb544ce298deba34947226a4",
            "spark-role": "driver",
            "sparkoperator.k8s.io/app-name": "my-spark-jdbc-2ea432f1",
            "sparkoperator.k8s.io/launched-by-spark-operator": "true",
            "sparkoperator.k8s.io/submission-id": "44343f6b-42ca-41d4-b01a-66052cc5c919",
        }
        driver_pod_name = "my-spark-jdbc-2ea432f1-driver"

        self.driver_pod = self._generate_pod(
            driver_pod_name, driver_pod_labels, PodPhases.running,
        )

        run_label_selector = self.runtime_handler._get_run_label_selector(
            self.project, self.run_uid
        )
        self.pod_label_selector = f"mlrun/class,{run_label_selector}"

    def test_list_sparkjob_resources(self):
        mocked_responses = self._mock_list_namespaced_crds([[self.completed_crd_dict]])
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.spark,
            expected_crds=mocked_responses[0]["items"],
            expected_pods=pods,
        )

    def test_delete_resources_completed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            [self.executor_pod, self.driver_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_custom_objects()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.completed_crd_dict["metadata"]["name"]],
            self.completed_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.driver_pod.metadata.name,
        )

    def test_delete_resources_running_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db)

        # nothing removed cause crd is running
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler, [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
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
        self._mock_delete_namespaced_custom_objects()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=10)

        # nothing removed cause grace period didn't pass
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler, [],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )

    def test_delete_resources_with_force(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            [self.executor_pod, self.driver_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_delete_namespaced_custom_objects()
        log = self._mock_read_namespaced_pod_log()
        self.runtime_handler.delete_resources(get_db(), db, force=True)
        self._assert_delete_namespaced_custom_objects(
            self.runtime_handler,
            [self.running_crd_dict["metadata"]["name"]],
            self.running_crd_dict["metadata"]["namespace"],
        )
        self._assert_list_namespaced_crds_calls(
            self.runtime_handler, len(list_namespaced_crds_calls),
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.running
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.driver_pod.metadata.name,
        )

    def test_monitor_run_completed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            [self.completed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            [self.executor_pod, self.driver_pod],
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
            self.runtime_handler, expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
        )
        self._assert_run_reached_state(
            db, self.project, self.run_uid, RunStates.completed
        )
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.driver_pod.metadata.name,
        )

    def test_monitor_run_failed_crd(self, db: Session, client: TestClient):
        list_namespaced_crds_calls = [
            [self.running_crd_dict],
            [self.failed_crd_dict],
        ]
        self._mock_list_namespaced_crds(list_namespaced_crds_calls)
        # for the get_logger_pods
        list_namespaced_pods_calls = [
            [self.executor_pod, self.driver_pod],
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
            self.runtime_handler, expected_number_of_list_crds_calls,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler,
            len(list_namespaced_pods_calls),
            self.pod_label_selector,
        )
        self._assert_run_reached_state(db, self.project, self.run_uid, RunStates.error)
        self._assert_run_logs(
            db, self.project, self.run_uid, log, self.driver_pod.metadata.name,
        )

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
                "namespace": get_k8s().resolve_namespace(),
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
    def _get_completed_crd_status(timestamp=None):
        return {
            "terminationTime": timestamp or "2020-10-05T21:17:11Z",
            "applicationState": {"state": "COMPLETED"},
        }

    @staticmethod
    def _get_failed_crd_status():
        return {
            "terminationTime": "2020-10-05T21:17:11Z",
            "applicationState": {"state": "FAILED"},
        }
