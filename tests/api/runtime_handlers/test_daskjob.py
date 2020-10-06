from mlrun.runtimes import RuntimeKinds
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestDaskjobRuntimeHandler(TestRuntimeHandlerBase):
    def test_list_resources(self):
        pods = self._mock_list_resources_pods()
        services = self._create_daskjob_service_mocks()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.dask, expected_pods=pods, expected_services=services,
        )

    @staticmethod
    def _mock_list_resources_pods():
        (
            scheduler_pod_dict,
            worker_pod_dict,
        ) = TestDaskjobRuntimeHandler._generate_pod_dicts()
        mocked_responses = TestDaskjobRuntimeHandler._mock_list_pods(
            [[scheduler_pod_dict, worker_pod_dict]]
        )
        return mocked_responses[0]

    @staticmethod
    def _generate_pod_dicts():
        scheduler_pod_dict = {
            "metadata": {
                "name": "mlrun-mydask-d7656bc1-0n4z9z",
                "labels": {
                    "app": "dask",
                    "dask.org/cluster-name": "mlrun-mydask-d7656bc1-0",
                    "dask.org/component": "scheduler",
                    "mlrun/class": "dask",
                    "mlrun/function": "mydask",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "user": "root",
                },
            },
            "status": {
                "conditions": [
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:35:15+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Initialized",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:36:20+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Ready",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:36:20+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "ContainersReady",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:35:15+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "PodScheduled",
                    },
                ],
                "container_statuses": [
                    {
                        "container_id": "docker://c24d68d4c71d8f61bed56aaf424dc7ffb86a9f59d7710afaa1e28f47bd466a5e",
                        "image": "mlrun/ml-models:0.5.1",
                        "image_id": "docker-pullable://mlrun/ml-models@sha256:07cc9e991dc603dbe100f4eae93d20f3ce53af1e6"
                        "d4af5191dbc66e6dfbce85b",
                        "last_state": {
                            "running": None,
                            "terminated": None,
                            "waiting": None,
                        },
                        "name": "base",
                        "ready": True,
                        "restart_count": 0,
                        "state": {
                            "running": {"started_at": "2020-08-18T00:36:19+00:00"},
                            "terminated": None,
                            "waiting": None,
                        },
                    }
                ],
                "host_ip": "172.31.6.138",
                "init_container_statuses": None,
                "message": None,
                "nominated_node_name": None,
                "phase": "Running",
                "pod_ip": "10.200.0.48",
                "qos_class": "BestEffort",
                "reason": None,
                "start_time": "2020-08-18T00:35:15+00:00",
            },
        }
        worker_pod_dict = {
            "metadata": {
                "name": "mlrun-mydask-d7656bc1-0pqbnc",
                "labels": {
                    "app": "dask",
                    "dask.org/cluster-name": "mlrun-mydask-d7656bc1-0",
                    "dask.org/component": "worker",
                    "mlrun/class": "dask",
                    "mlrun/function": "mydask",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "user": "root",
                },
            },
            "status": {
                "conditions": [
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:36:21+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Initialized",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:36:24+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Ready",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:36:24+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "ContainersReady",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T00:36:21+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "PodScheduled",
                    },
                ],
                "container_statuses": [
                    {
                        "container_id": "docker://18f75b15b9fbf0ed9136d9ec7f14cf1d62dbfa078877f89847fa346fe09ff574",
                        "image": "mlrun/ml-models:0.5.1",
                        "image_id": "docker-pullable://mlrun/ml-models@sha256:07cc9e991dc603dbe100f4eae93d20f3ce53af1e6"
                        "d4af5191dbc66e6dfbce85b",
                        "last_state": {
                            "running": None,
                            "terminated": None,
                            "waiting": None,
                        },
                        "name": "base",
                        "ready": True,
                        "restart_count": 0,
                        "state": {
                            "running": {"started_at": "2020-08-18T00:36:23+00:00"},
                            "terminated": None,
                            "waiting": None,
                        },
                    }
                ],
                "host_ip": "172.31.6.138",
                "init_container_statuses": None,
                "message": None,
                "nominated_node_name": None,
                "phase": "Running",
                "pod_ip": "10.200.0.51",
                "qos_class": "BestEffort",
                "reason": None,
                "start_time": "2020-08-18T00:36:21+00:00",
            },
        }
        return scheduler_pod_dict, worker_pod_dict

    @staticmethod
    def _create_daskjob_service_mocks():
        service_dict = {
            "metadata": {
                "name": "mlrun-mydask-d7656bc1-0",
                "labels": {
                    "app": "dask",
                    "dask.org/cluster-name": "mlrun-mydask-d7656bc1-0",
                    "dask.org/component": "scheduler",
                    "mlrun/class": "dask",
                    "mlrun/function": "mydask",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "user": "root",
                },
            },
        }
        return TestDaskjobRuntimeHandler._mock_list_services([service_dict])
