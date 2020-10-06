from mlrun.config import config
from mlrun.runtimes import RuntimeKinds
from mlrun.runtimes.constants import MPIJobCRDVersions
from tests.api.runtime_handlers.base import TestRuntimeHandlerBase


class TestMPIjobRuntimeHandler(TestRuntimeHandlerBase):
    def test_list_mpijob_resources(self, k8s_helper_mock):
        config.mpijob_crd_version = MPIJobCRDVersions.v1
        crds = self._mock_list_mpijob_crds(k8s_helper_mock)

        # there's currently a bug (fix was merged but not released https://github.com/kubeflow/mpi-operator/pull/271)
        # that causes mpijob's pods to not being labels with the given (MLRun's) labels - this prevents list resources
        # from finding the pods, so we're simulating the same thing here
        k8s_helper_mock.list_pods.return_value = []
        self._assert_runtime_handler_list_resources(
            RuntimeKinds.mpijob, k8s_helper_mock, expected_crds=crds
        )

    @staticmethod
    def _mock_list_mpijob_crds(k8s_helper_mock):
        crd_dict = {
            "metadata": {
                "name": "train-eaf63df8",
                "labels": {
                    "mlrun/class": "mpijob",
                    "mlrun/function": "trainer",
                    "mlrun/name": "train",
                    "mlrun/project": "cat-and-dog-servers",
                    "mlrun/scrape_metrics": "False",
                    "mlrun/tag": "latest",
                    "mlrun/uid": "9401e4b27f004c6ba750d3e936f1fccb",
                },
            },
            "status": {
                "completionTime": "2020-08-18T01:23:54Z",
                "conditions": [
                    {
                        "lastTransitionTime": "2020-08-18T01:21:15Z",
                        "lastUpdateTime": "2020-08-18T01:21:15Z",
                        "message": "MPIJob default-tenant/train-eaf63df8 is created.",
                        "reason": "MPIJobCreated",
                        "status": "True",
                        "type": "Created",
                    },
                    {
                        "lastTransitionTime": "2020-08-18T01:21:23Z",
                        "lastUpdateTime": "2020-08-18T01:21:23Z",
                        "message": "MPIJob default-tenant/train-eaf63df8 is running.",
                        "reason": "MPIJobRunning",
                        "status": "False",
                        "type": "Running",
                    },
                    {
                        "lastTransitionTime": "2020-08-18T01:23:54Z",
                        "lastUpdateTime": "2020-08-18T01:23:54Z",
                        "message": "MPIJob default-tenant/train-eaf63df8 successfully completed.",
                        "reason": "MPIJobSucceeded",
                        "status": "True",
                        "type": "Succeeded",
                    },
                ],
                "replicaStatuses": {"Launcher": {"succeeded": 1}, "Worker": {}},
                "startTime": "2020-08-18T01:21:15Z",
            },
        }
        return TestMPIjobRuntimeHandler._mock_list_crds(k8s_helper_mock, [crd_dict])
