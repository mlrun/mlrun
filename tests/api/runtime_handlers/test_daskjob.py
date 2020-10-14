from kubernetes import client

from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes.constants import PodPhases
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestDaskjobRuntimeHandler(TestRuntimeHandlerBase):
    def custom_setup(self):
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

        self.scheduler_pod = self._generate_pod(
            scheduler_pod_name, scheduler_pod_labels, PodPhases.running,
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

        self.worker_pod = self._generate_pod(
            worker_pod_name, worker_pod_labels, PodPhases.running,
        )

    def test_list_resources(self):
        pods = self._mock_list_resources_pods()
        services = self._create_daskjob_service_mocks()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.dask, expected_pods=pods, expected_services=services,
        )

    def _mock_list_resources_pods(self):
        mocked_responses = TestDaskjobRuntimeHandler._mock_list_namespaces_pods(
            [[self.scheduler_pod, self.worker_pod]]
        )
        return mocked_responses[0].items

    @staticmethod
    def _create_daskjob_service_mocks():
        name = "mlrun-mydask-d7656bc1-0"
        labels = {
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
        metadata = client.V1ObjectMeta(name=name, labels=labels)
        service = client.V1Service(metadata=metadata)
        return TestDaskjobRuntimeHandler._mock_list_services([service])
