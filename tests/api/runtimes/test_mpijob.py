import typing
import unittest.mock

from kubernetes import client as k8s_client

import mlrun.runtimes.pod
from mlrun import code_to_function
from mlrun.api.utils.singletons.k8s import get_k8s
from tests.api.runtimes.base import TestRuntimeBase


class TestMpiJob(TestRuntimeBase):
    def custom_setup(self):
        self.runtime_kind = "mpijob"
        self.code_handler = "test_func"

    def _get_pod(self):
        return k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(
                labels={
                    "kind": "mpijob",
                    "owner": "tester",
                    "v3io_user": "tester",
                },
                name=self.name,
            ),
        )

    def _mock_list_pods(self, pods=None):
        if pods is None:
            pods = [self._get_pod()]
        get_k8s().list_pods = unittest.mock.Mock(return_value=pods)

    def _generate_runtime(
        self, kind=None, labels=None
    ) -> typing.Union[mlrun.runtimes.MpiRuntimeV1, mlrun.runtimes.MpiRuntimeV1Alpha1]:
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind=kind or self.runtime_kind,
            image=self.image_name,
            description="test mpijob",
            labels=labels,
        )
        return runtime


class TestMpiV1Runtime(TestMpiJob):
    def custom_setup(self):
        super(TestMpiV1Runtime, self).custom_setup()
        self.name = "test-mpi-v1"

    def custom_teardown(self):
        pass

    def test_run_state_completion(self):
        self._mock_list_pods()
        mpijob_function = self._generate_runtime(self.runtime_kind)
        mpijob_function.spec.replicas = 4
        mpijob_function.deploy()
        mpijob_function.run()
