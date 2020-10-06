from mlrun.runtimes import RuntimeKinds
from tests.runtimes.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def test_list_resources(self, k8s_helper_mock):
        pods = self._mock_list_resources_pods(k8s_helper_mock)
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, k8s_helper_mock, expected_pods=pods
        )

    @staticmethod
    def _mock_list_resources_pods(k8s_helper_mock):
        pod_dict = TestKubejobRuntimeHandler._generate_pod_dict()
        mocked_responses = k8s_helper_mock.mock_list_pods([[pod_dict]])
        return mocked_responses[0]

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
            "conditions": [
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "Initialized",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": "containers with unready status: [base]",
                    "reason": "ContainersNotReady",
                    "status": "False",
                    "type": "Ready",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": "containers with unready status: [base]",
                    "reason": "ContainersNotReady",
                    "status": "False",
                    "type": "ContainersReady",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "PodScheduled",
                },
            ],
            "container_statuses": [
                {
                    "container_id": None,
                    "image": "docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80/mlrun/func-default-h"
                    "edi-simple-func-latest",
                    "image_id": "",
                    "last_state": {
                        "running": None,
                        "terminated": None,
                        "waiting": None,
                    },
                    "name": "base",
                    "ready": False,
                    "restart_count": 0,
                    "state": {
                        "running": None,
                        "terminated": None,
                        "waiting": {"message": None, "reason": "ContainerCreating"},
                    },
                }
            ],
            "host_ip": "172.31.4.201",
            "init_container_statuses": None,
            "message": None,
            "nominated_node_name": None,
            "phase": "Pending",
            "pod_ip": None,
            "qos_class": "BestEffort",
            "reason": None,
            "start_time": "2020-10-6T3:0:50+00:00",
        }

    @staticmethod
    def _get_running_pod_status():
        return {
            "conditions": [
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "Initialized",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:52+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "Ready",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:52+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "ContainersReady",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "PodScheduled",
                },
            ],
            "container_statuses": [
                {
                    "container_id": "docker://94a90870a6432d3140da821b87ae91980d21af2c000988fcb8687640a5f29886",
                    "image": "docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80/mlrun/func-default-h"
                    "edi-simple-func-latest:latest",
                    "image_id": "docker-pullable://docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80"
                    "/mlrun/func-default-hedi-simple-func-latest@sha256:29a8b029b0b10b87a48c71a3161515d27f5"
                    "65ec52f5cca04f01c1cde2e875152",
                    "last_state": {
                        "running": None,
                        "terminated": None,
                        "waiting": None,
                    },
                    "name": "base",
                    "ready": True,
                    "restart_count": 0,
                    "state": {
                        "running": {"started_at": "2020-10-6T3:0:51+00:00"},
                        "terminated": None,
                        "waiting": None,
                    },
                }
            ],
            "host_ip": "172.31.4.201",
            "init_container_statuses": None,
            "message": None,
            "nominated_node_name": None,
            "phase": "Running",
            "pod_ip": "10.200.0.51",
            "qos_class": "BestEffort",
            "reason": None,
            "start_time": "2020-10-6T3:0:50+00:00",
        }

    @staticmethod
    def _get_completed_pod_status():
        return {
            "conditions": [
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": None,
                    "reason": "PodCompleted",
                    "status": "True",
                    "type": "Initialized",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:1:8+00:00",
                    "message": None,
                    "reason": "PodCompleted",
                    "status": "False",
                    "type": "Ready",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:1:8+00:00",
                    "message": None,
                    "reason": "PodCompleted",
                    "status": "False",
                    "type": "ContainersReady",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T3:0:50+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "PodScheduled",
                },
            ],
            "container_statuses": [
                {
                    "container_id": "docker://94a90870a6432d3140da821b87ae91980d21af2c000988fcb8687640a5f29886",
                    "image": "docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80/mlrun/func-default-h"
                    "edi-simple-func-latest:latest",
                    "image_id": "docker-pullable://docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80"
                    "/mlrun/func-default-hedi-simple-func-latest@sha256:29a8b029b0b10b87a48c71a3161515d27f5"
                    "65ec52f5cca04f01c1cde2e875152",
                    "last_state": {
                        "running": None,
                        "terminated": None,
                        "waiting": None,
                    },
                    "name": "base",
                    "ready": False,
                    "restart_count": 0,
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
            "host_ip": "172.31.4.201",
            "init_container_statuses": None,
            "message": None,
            "nominated_node_name": None,
            "phase": "Succeeded",
            "pod_ip": "10.200.0.51",
            "qos_class": "BestEffort",
            "reason": None,
            "start_time": "2020-10-6T3:0:50+00:00",
        }

    @staticmethod
    def _get_failed_pod_status():
        return {
            "conditions": [
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T2:59:34+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "Initialized",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T2:59:53+00:00",
                    "message": "containers with unready status: [base]",
                    "reason": "ContainersNotReady",
                    "status": "False",
                    "type": "Ready",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T2:59:53+00:00",
                    "message": "containers with unready status: [base]",
                    "reason": "ContainersNotReady",
                    "status": "False",
                    "type": "ContainersReady",
                },
                {
                    "last_probe_time": None,
                    "last_transition_time": "2020-10-6T2:59:34+00:00",
                    "message": None,
                    "reason": None,
                    "status": "True",
                    "type": "PodScheduled",
                },
            ],
            "container_statuses": [
                {
                    "container_id": "docker://ec259b0c68d9bc981964859ecac3d2da107b38da4fa7ca3df3c3eedb61bfb47e",
                    "image": "docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80/mlrun/func-default-h"
                    "edi-simple-func-latest:latest",
                    "image_id": "docker-pullable://docker-registry.default-tenant.app.hedingber-30-2.iguazio-cd2.com:80"
                    "/mlrun/func-default-hedi-simple-func-latest@sha256:29a8b029b0b10b87a48c71a3161515d27f5"
                    "65ec52f5cca04f01c1cde2e875152",
                    "last_state": {
                        "running": None,
                        "terminated": None,
                        "waiting": None,
                    },
                    "name": "base",
                    "ready": False,
                    "restart_count": 0,
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
            "host_ip": "172.31.4.201",
            "init_container_statuses": None,
            "message": None,
            "nominated_node_name": None,
            "phase": "Failed",
            "pod_ip": "10.200.0.51",
            "qos_class": "BestEffort",
            "reason": None,
            "start_time": "2020-10-6T2:59:34+00:00",
        }
