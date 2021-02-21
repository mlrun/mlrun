import os
import unittest
from tests.api.runtimes.base import TestRuntimeBase
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import mlrun
from mlrun.platforms import auto_mount
from dask import distributed
from mlrun import mlconf
from mlrun.runtimes.utils import generate_resources


class TestDaskRuntime(TestRuntimeBase):
    def _mock_dask_cluster(self):
        patcher = unittest.mock.patch('dask_kubernetes.KubeCluster')
        self.kube_cluster_mock = patcher.start()
        self.kube_cluster_mock.return_value.name = self.name
        self.kube_cluster_mock.return_value.scheduler_address = self.scheduler_address

        class MockPort:
            def __init__(self, port):
                self.node_port = port

        # 1st port is client port, 2nd port is dashboard port, both apply to the ingress
        self.kube_cluster_mock.return_value.scheduler.service.spec.ports = [MockPort(1234), MockPort(5678)]

        distributed.Client = unittest.mock.Mock()

    def custom_setup(self):
        self.name = "test-dask-cluster"
        # For dask it is /function instead of /name
        self.function_name_label = "mlrun/function"
        self.v3io_access_key = "1111-2222-3333-4444"
        self.v3io_user = "test-user"
        self.scheduler_address = "http://1.2.3.4"

        self._mock_dask_cluster()

    def custom_teardown(self):
        unittest.mock.patch.stopall()

    def _get_pod_creation_args(self):
        args, _ = self.kube_cluster_mock.call_args
        return args[0]

    def _get_namespace_arg(self):
        _, kwargs = self.kube_cluster_mock.call_args
        return kwargs["namespace"]

    def _generate_runtime(self):
        # This is following the steps in
        # https://docs.mlrun.org/en/latest/runtimes/dask-mlrun.html#set-up-the-enviroment
        mlconf.remote_host = "http://remote_host"
        os.environ["V3IO_USERNAME"] = self.v3io_user

        mlrun.set_environment(project=self.project,
                              access_key=self.v3io_access_key,
                              artifact_path=self.artifact_path)
        dask_cluster = mlrun.new_function(
            self.name, project=self.project, kind='dask', image=self.image_name
        )

        dask_cluster.apply(auto_mount())

        dask_cluster.spec.min_replicas = 1
        dask_cluster.spec.max_replicas = 4

        dask_cluster.spec.remote = True
        dask_cluster.spec.service_type = "NodePort"

        return dask_cluster

    def test_dask_runtime(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        expected_requests = generate_resources(mem="2G", cpu=3)
        runtime.with_requests(
            mem=expected_requests["memory"], cpu=expected_requests["cpu"]
        )
        _ = runtime.client

        self.kube_cluster_mock.assert_called_once()

        self._assert_pod_create_called(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
            expected_requests=expected_requests,
        )
        self._assert_v3io_mount_configured(self.v3io_user, self.v3io_access_key)

