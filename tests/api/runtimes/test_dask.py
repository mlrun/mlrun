import base64
import json
import os
import unittest

from dask import distributed
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
from mlrun import mlconf
from mlrun.platforms import auto_mount
from mlrun.runtimes.utils import generate_resources
from tests.api.runtimes.base import TestRuntimeBase


class TestDaskRuntime(TestRuntimeBase):
    def _mock_dask_cluster(self):
        patcher = unittest.mock.patch("dask_kubernetes.KubeCluster")
        self.kube_cluster_mock = patcher.start()
        self.kube_cluster_mock.return_value.name = self.name
        self.kube_cluster_mock.return_value.scheduler_address = self.scheduler_address

        class MockPort:
            def __init__(self, port):
                self.node_port = port

        # 1st port is client port, 2nd port is dashboard port, both apply to the ingress
        self.kube_cluster_mock.return_value.scheduler.service.spec.ports = [
            MockPort(1234),
            MockPort(5678),
        ]

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
        return self._get_worker_pod_creation_args()

    def _get_worker_pod_creation_args(self):
        args, _ = self.kube_cluster_mock.call_args
        return args[0]

    def _get_scheduler_pod_creation_args(self):
        _, kwargs = self.kube_cluster_mock.call_args
        return kwargs["scheduler_pod_template"]

    def _get_namespace_arg(self):
        _, kwargs = self.kube_cluster_mock.call_args
        return kwargs["namespace"]

    def _generate_runtime(self):
        # This is following the steps in
        # https://docs.mlrun.org/en/latest/runtimes/dask-mlrun.html#set-up-the-environment
        mlconf.remote_host = "http://remote_host"
        os.environ["V3IO_USERNAME"] = self.v3io_user
        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key
        mlconf.default_project = self.project
        mlconf.artifact_path = self.artifact_path

        dask_cluster = mlrun.new_function(
            self.name, project=self.project, kind="dask", image=self.image_name
        )

        dask_cluster.apply(auto_mount())

        dask_cluster.spec.min_replicas = 1
        dask_cluster.spec.max_replicas = 4

        dask_cluster.spec.remote = True
        dask_cluster.spec.service_type = "NodePort"

        return dask_cluster

    def _assert_scheduler_pod_args(self,):
        scheduler_pod = self._get_scheduler_pod_creation_args()
        scheduler_container_spec = scheduler_pod.spec.containers[0]
        assert scheduler_container_spec.args == ["dask-scheduler"]

    def _assert_pods_resources(
        self,
        expected_worker_requests,
        expected_worker_limits,
        expected_scheduler_requests,
        expected_scheduler_limits,
    ):
        worker_pod = self._get_pod_creation_args()
        worker_container_spec = worker_pod.spec.containers[0]
        self._assert_container_resources(
            worker_container_spec, expected_worker_limits, expected_worker_requests
        )
        scheduler_pod = self._get_scheduler_pod_creation_args()
        scheduler_container_spec = scheduler_pod.spec.containers[0]
        self._assert_container_resources(
            scheduler_container_spec,
            expected_scheduler_limits,
            expected_scheduler_requests,
        )

    def test_dask_runtime(self, db: Session, client: TestClient):
        runtime: mlrun.runtimes.DaskCluster = self._generate_runtime()

        _ = runtime.client

        self.kube_cluster_mock.assert_called_once()

        self._assert_pod_creation_config(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
        )
        self._assert_v3io_mount_or_creds_configured(
            self.v3io_user, self.v3io_access_key
        )
        self._assert_scheduler_pod_args()

    def test_dask_runtime_with_resources(self, db: Session, client: TestClient):
        runtime: mlrun.runtimes.DaskCluster = self._generate_runtime()

        expected_requests = generate_resources(mem="2G", cpu=3)
        runtime.with_requests(
            mem=expected_requests["memory"], cpu=expected_requests["cpu"]
        )
        gpu_type = "nvidia.com/gpu"
        expected_gpus = 2
        expected_scheduler_limits = generate_resources(
            mem="4G", cpu=5, gpus=expected_gpus, gpu_type=gpu_type
        )
        expected_worker_limits = generate_resources(
            mem="4G", cpu=5, gpus=expected_gpus, gpu_type=gpu_type
        )
        runtime.with_scheduler_limits(
            mem=expected_scheduler_limits["memory"],
            cpu=expected_scheduler_limits["cpu"],
        )
        runtime.with_worker_limits(
            mem=expected_worker_limits["memory"], cpu=expected_worker_limits["cpu"],
        )
        runtime.gpus(expected_gpus, gpu_type)
        _ = runtime.client

        self.kube_cluster_mock.assert_called_once()

        self._assert_pod_creation_config(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
        )
        self._assert_v3io_mount_or_creds_configured(
            self.v3io_user, self.v3io_access_key
        )
        self._assert_pods_resources(
            expected_requests,
            expected_worker_limits,
            expected_requests,
            expected_scheduler_limits,
        )

    def test_dask_with_node_selection(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        node_name = "some-node-name"
        runtime.with_node_selection(node_name)
        node_selector = {
            "label-a": "val1",
            "label-2": "val2",
        }
        runtime.with_node_selection(node_selector=node_selector)
        affinity = self._generate_affinity()
        runtime.with_node_selection(affinity=affinity)
        _ = runtime.client

        self.kube_cluster_mock.assert_called_once()

        self._assert_pod_creation_config(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
            expected_node_name=node_name,
            expected_node_selector=node_selector,
            expected_affinity=affinity,
        )

    def test_dask_with_priority_class_name(self, db: Session, client: TestClient):
        default_priority_class_name = "default-priority"
        mlrun.mlconf.default_function_priority_class_name = default_priority_class_name
        mlrun.mlconf.valid_function_priority_class_names = default_priority_class_name
        runtime = self._generate_runtime()

        _ = runtime.client

        self.kube_cluster_mock.assert_called_once()

        self._assert_pod_creation_config(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
            expected_priority_class_name=default_priority_class_name,
        )

        runtime = self._generate_runtime()
        medium_priority_class_name = "medium-priority"
        mlrun.mlconf.valid_function_priority_class_names = ",".join(
            [default_priority_class_name, medium_priority_class_name]
        )
        runtime.with_priority_class(medium_priority_class_name)

        _ = runtime.client

        self._assert_pod_creation_config(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
            expected_priority_class_name=medium_priority_class_name,
        )

    def test_dask_with_default_node_selector(self, db: Session, client: TestClient):
        node_selector = {
            "label-a": "val1",
            "label-2": "val2",
        }
        mlrun.mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        runtime = self._generate_runtime()
        _ = runtime.client

        self.kube_cluster_mock.assert_called_once()

        self._assert_pod_creation_config(
            expected_runtime_class_name="dask",
            assert_create_pod_called=False,
            assert_namespace_env_variable=False,
            expected_node_selector=node_selector,
        )
