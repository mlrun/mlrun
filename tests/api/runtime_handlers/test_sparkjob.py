from mlrun.runtimes import RuntimeKinds
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestSparkjobRuntimeHandler(TestRuntimeHandlerBase):
    def test_list_sparkjob_resources(self):
        crds = self._mock_list_sparkjob_crds()
        pods = self._mock_list_resources_pods()
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.spark, expected_crds=crds, expected_pods=pods
        )

    @staticmethod
    def _mock_list_sparkjob_crds():
        crd_dict = {
            "metadata": {
                "name": "my-spark-jdbc-2ea432f1",
                "labels": {
                    "mlrun/class": "spark",
                    "mlrun/function": "my-spark-jdbc",
                    "mlrun/name": "my-spark-jdbc",
                    "mlrun/project": "default",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": "b532ba206a1649da9925d340d6f97f7a",
                },
            },
            "status": {
                "applicationState": {"state": "RUNNING"},
                "driverInfo": {
                    "podName": "my-spark-jdbc-2ea432f1-driver",
                    "webUIAddress": "10.197.111.54:0",
                    "webUIPort": 4040,
                    "webUIServiceName": "my-spark-jdbc-2ea432f1-ui-svc",
                },
                "executionAttempts": 2,
                "executorState": {
                    "my-spark-jdbc-2ea432f1-1597760338437-exec-1": "RUNNING"
                },
                "sparkApplicationId": "spark-12f88a73cb544ce298deba34947226a4",
                "submissionAttempts": 1,
                "submissionID": "44343f6b-42ca-41d4-b01a-66052cc5c919",
                "submissionTime": "2020-08-18T14:19:16Z",
                "terminationTime": None,
            },
        }
        return TestSparkjobRuntimeHandler._mock_list_crds([crd_dict])

    @staticmethod
    def _mock_list_resources_pods():
        (
            executor_pod_dict,
            driver_pod_dict,
        ) = TestSparkjobRuntimeHandler._generate_pod_dicts()
        mocked_responses = TestSparkjobRuntimeHandler._mock_list_namespaces_pods(
            [[executor_pod_dict, driver_pod_dict]]
        )
        return mocked_responses[0]

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
                "conditions": [
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:25+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Initialized",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:28+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Ready",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:28+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "ContainersReady",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:25+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "PodScheduled",
                    },
                ],
                "container_statuses": [
                    {
                        "container_id": "docker://de6c8574b113b1200bae56918e77b4f8f344f18741d8f53cdb5eab5c55f6c16a",
                        "image": "iguazio/spark-app:2.10_b59_20200813105414",
                        "image_id": "docker://sha256:251e43e69e8449dc45883ad4e5d3cf785068fa86852335d69e56b605c6bd03"
                        "0b",
                        "last_state": {
                            "running": None,
                            "terminated": None,
                            "waiting": None,
                        },
                        "name": "executor",
                        "ready": True,
                        "restart_count": 0,
                        "state": {
                            "running": {"started_at": "2020-08-18T14:19:28+00:00"},
                            "terminated": None,
                            "waiting": None,
                        },
                    }
                ],
                "host_ip": "172.31.7.224",
                "init_container_statuses": None,
                "message": None,
                "nominated_node_name": None,
                "phase": "Running",
                "pod_ip": "10.200.0.53",
                "qos_class": "Burstable",
                "reason": None,
                "start_time": "2020-08-18T14:19:25+00:00",
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
                "conditions": [
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:08+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Initialized",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:17+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "Ready",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:17+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "ContainersReady",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-18T14:19:08+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "PodScheduled",
                    },
                ],
                "container_statuses": [
                    {
                        "container_id": "docker://916268e7baf76e95fc3a8b79227c4807e4f421004e6674649faaa0540d6cad29",
                        "image": "iguazio/spark-app:2.10_b59_20200813105414",
                        "image_id": "docker://sha256:251e43e69e8449dc45883ad4e5d3cf785068fa86852335d69e56b605c6bd030b",
                        "last_state": {
                            "running": None,
                            "terminated": None,
                            "waiting": None,
                        },
                        "name": "spark-kubernetes-driver",
                        "ready": True,
                        "restart_count": 0,
                        "state": {
                            "running": {"started_at": "2020-08-18T14:19:16+00:00"},
                            "terminated": None,
                            "waiting": None,
                        },
                    }
                ],
                "host_ip": "172.31.7.224",
                "init_container_statuses": None,
                "message": None,
                "nominated_node_name": None,
                "phase": "Running",
                "pod_ip": "10.200.0.52",
                "qos_class": "Burstable",
                "reason": None,
                "start_time": "2020-08-18T14:19:08+00:00",
            },
        }
        return executor_pod_dict, driver_pod_dict
