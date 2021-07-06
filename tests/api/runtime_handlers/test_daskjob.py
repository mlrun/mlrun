from fastapi.testclient import TestClient
from kubernetes import client
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db
from mlrun.runtimes import RuntimeKinds, get_runtime_handler
from mlrun.runtimes.constants import PodPhases
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestDaskjobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
        self.runtime_handler = get_runtime_handler(RuntimeKinds.dask)
        self.runtime_handler.wait_for_deletion_interval = 0

        # initializing them here to save space in tests
        scheduler_pod_labels = {
            "app": "dask",
            "dask.org/cluster-name": "mlrun-mydask-d7656bc1-0",
            "dask.org/component": "scheduler",
            "mlrun/class": "dask",
            "mlrun/function": "mydask",
            "mlrun/project": "default",
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "user": "root",
        }
        scheduler_pod_name = "mlrun-mydask-d7656bc1-0n4z9z"

        self.running_scheduler_pod = self._generate_pod(
            scheduler_pod_name, scheduler_pod_labels, PodPhases.running,
        )
        self.completed_scheduler_pod = self._generate_pod(
            scheduler_pod_name, scheduler_pod_labels, PodPhases.succeeded,
        )

        worker_pod_labels = {
            "app": "dask",
            "dask.org/cluster-name": "mlrun-mydask-d7656bc1-0",
            "dask.org/component": "worker",
            "mlrun/class": "dask",
            "mlrun/function": "mydask",
            "mlrun/project": "default",
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "user": "root",
        }
        worker_pod_name = "mlrun-mydask-d7656bc1-0pqbnc"

        self.running_worker_pod = self._generate_pod(
            worker_pod_name, worker_pod_labels, PodPhases.running,
        )
        self.completed_worker_pod = self._generate_pod(
            worker_pod_name, worker_pod_labels, PodPhases.succeeded,
        )

        service_name = "mlrun-mydask-d7656bc1-0"
        service_labels = {
            "app": "dask",
            "dask.org/cluster-name": "mlrun-mydask-d7656bc1-0",
            "dask.org/component": "scheduler",
            "mlrun/class": "dask",
            "mlrun/function": "mydask",
            "mlrun/project": "default",
            "mlrun/scrape_metrics": "False",
            "mlrun/tag": "latest",
            "user": "root",
        }

        self.cluster_service = self._generate_service(service_name, service_labels)

    def test_list_resources(self, db: Session, client: TestClient):
        pods = self._mock_list_resources_pods()
        services = self._mock_list_services([self.cluster_service])
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.dask, expected_pods=pods, expected_services=services,
        )

    def test_delete_resources_completed_cluster(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.completed_worker_pod, self.completed_scheduler_pod],
            # additional time for wait for pods deletion - simulate pods not removed yet
            [self.completed_worker_pod, self.completed_scheduler_pod],
            # additional time for wait for pods deletion - simulate pods gone
            [],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_list_services([self.cluster_service])
        self._mock_delete_namespaced_pods()
        self._mock_delete_namespaced_services()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods(
            [
                self.completed_worker_pod.metadata.name,
                self.completed_scheduler_pod.metadata.name,
            ],
            self.completed_scheduler_pod.metadata.namespace,
        )
        self._assert_delete_namespaced_services(
            [self.completed_scheduler_pod.metadata.labels.get("dask.org/cluster-name")],
            self.completed_scheduler_pod.metadata.namespace,
        )
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_delete_resources_running_cluster(self, db: Session, client: TestClient):
        list_namespaced_pods_calls = [
            [self.running_worker_pod, self.running_scheduler_pod],
        ]
        self._mock_list_namespaced_pods(list_namespaced_pods_calls)
        self._mock_list_services([self.cluster_service])
        self._mock_delete_namespaced_pods()
        self._mock_delete_namespaced_services()
        self.runtime_handler.delete_resources(get_db(), db, grace_period=0)
        self._assert_delete_namespaced_pods([])
        self._assert_delete_namespaced_services([])
        self._assert_list_namespaced_pods_calls(
            self.runtime_handler, len(list_namespaced_pods_calls)
        )

    def test_monitor_run(self, db: Session, client: TestClient):
        """
        There's no real monitoring for dask (see mlrun.runtimes.daskjob.DaskRuntimeHandler.monitor_run for explanation
        why) this test is only to have coverage for the piece of code there
        """
        self.runtime_handler.monitor_runs(get_db(), db, None)

    def _mock_list_resources_pods(self):
        mocked_responses = TestDaskjobRuntimeHandler._mock_list_namespaced_pods(
            [[self.running_scheduler_pod, self.running_worker_pod]]
        )
        return mocked_responses[0].items

    @staticmethod
    def _generate_service(name, labels):
        metadata = client.V1ObjectMeta(name=name, labels=labels)
        return client.V1Service(metadata=metadata)
