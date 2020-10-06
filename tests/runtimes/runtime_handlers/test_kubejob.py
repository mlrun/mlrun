from mlrun.runtimes import RuntimeKinds
from tests.runtimes.runtime_handlers.base import TestRuntimeHandlerBase


class TestKubejobRuntimeHandler(TestRuntimeHandlerBase):
    def test_list_resources(self, k8s_helper_mock):
        pods = self._mock_list_kubejob_pods(k8s_helper_mock)
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.job, k8s_helper_mock, expected_pods=pods
        )

    @staticmethod
    def _mock_list_kubejob_pods(k8s_helper_mock):
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
            "status": {
                "conditions": [
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-17T18:08:23+00:00",
                        "message": None,
                        "reason": "PodCompleted",
                        "status": "True",
                        "type": "Initialized",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-17T18:08:47+00:00",
                        "message": None,
                        "reason": "PodCompleted",
                        "status": "False",
                        "type": "Ready",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-17T18:08:47+00:00",
                        "message": None,
                        "reason": "PodCompleted",
                        "status": "False",
                        "type": "ContainersReady",
                    },
                    {
                        "last_probe_time": None,
                        "last_transition_time": "2020-08-17T18:08:23+00:00",
                        "message": None,
                        "reason": None,
                        "status": "True",
                        "type": "PodScheduled",
                    },
                ],
                "container_statuses": [
                    {
                        "container_id": "docker://c00c36dc9a702508c76b6074f2c2fa3e569daaf13f5a72931804da04a6e96987",
                        "image": "docker-registry.default-tenant.app.hedingber-210-1.iguazio-cd0.com:80/mlrun/func-defa"
                        "ult-my-trainer-latest:latest",
                        "image_id": "docker-pullable://docker-registry.default-tenant.app.hedingber-210-1.iguazio-cd0.c"
                        "om:80/mlrun/func-default-my-trainer-latest@sha256:d23c93a997fa5ab89d899bf1bf1cb97f"
                        "a50697a74c61927c1df3266340076efc",
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
                                "container_id": "docker://c00c36dc9a702508c76b6074f2c2fa3e569daaf13f5a72931804da04a6e96"
                                "987",
                                "exit_code": 0,
                                "finished_at": "2020-08-17T18:08:47+00:00",
                                "message": None,
                                "reason": "Completed",
                                "signal": None,
                                "started_at": "2020-08-17T18:08:42+00:00",
                            },
                            "waiting": None,
                        },
                    }
                ],
                "host_ip": "172.31.6.138",
                "init_container_statuses": None,
                "message": None,
                "nominated_node_name": None,
                "phase": "Succeeded",
                "pod_ip": "10.200.0.48",
                "qos_class": "BestEffort",
                "reason": None,
                "start_time": "2020-08-17T18:08:23+00:00",
            },
        }
        return k8s_helper_mock.mock_list_pods([pod_dict])
