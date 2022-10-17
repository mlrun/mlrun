# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import base64
import json
import os
import unittest
import unittest.mock

from dask import distributed
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.endpoints.functions
import mlrun.api.schemas
from mlrun import mlconf
from mlrun.platforms import auto_mount
from mlrun.runtimes.utils import generate_resources
from tests.api.conftest import K8sSecretsMock
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

    def _get_create_pod_namespace_arg(self):
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

    def _assert_scheduler_pod_args(
        self,
    ):
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

    def assert_security_context(
        self,
        security_context=None,
        worker=True,
        scheduler=True,
    ):
        if worker:
            pod = self._get_pod_creation_args()
            assert pod.spec.security_context == (
                security_context or {}
            ), "Failed asserting security context in worker pod"
        if scheduler:
            scheduler_pod = self._get_scheduler_pod_creation_args()
            assert scheduler_pod.spec.security_context == (
                security_context or {}
            ), "Failed asserting security context in scheduler pod"

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

    def test_dask_runtime_with_resources_patch(self, db: Session, client: TestClient):
        runtime: mlrun.runtimes.DaskCluster = self._generate_runtime()
        runtime.with_scheduler_requests(mem="2G", cpu="3")
        runtime.with_worker_requests(mem="2G", cpu="3")
        gpu_type = "nvidia.com/gpu"

        runtime.with_scheduler_limits(mem="4G", cpu="5", gpu_type=gpu_type, gpus=2)
        runtime.with_worker_limits(mem="4G", cpu="5", gpu_type=gpu_type, gpus=2)

        runtime.with_scheduler_limits(gpus=3)  # default patch = False
        runtime.with_scheduler_requests(cpu="4", patch=True)

        runtime.with_worker_limits(cpu="10", patch=True)
        runtime.with_worker_requests(mem="3G")  # default patch = False
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
            expected_worker_requests={
                "memory": "3G",
            },
            expected_worker_limits={"memory": "4G", "cpu": "10", "nvidia.com/gpu": 2},
            expected_scheduler_requests={
                "memory": "2G",
                "cpu": "4",
            },
            expected_scheduler_limits={"nvidia.com/gpu": 3},
        )

    def test_dask_runtime_with_resources(self, db: Session, client: TestClient):
        runtime: mlrun.runtimes.DaskCluster = self._generate_runtime()

        expected_requests = generate_resources(mem="2G", cpu=3)
        runtime.with_scheduler_requests(
            mem=expected_requests["memory"], cpu=expected_requests["cpu"]
        )
        runtime.with_worker_requests(
            mem=expected_requests["memory"], cpu=expected_requests["cpu"]
        )
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
            mem=expected_worker_limits["memory"],
            cpu=expected_worker_limits["cpu"],
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

    def test_dask_runtime_without_specifying_resources(
        self, db: Session, client: TestClient
    ):
        for test_case in [
            {
                # when are not defaults defined
                "default_function_pod_resources": {
                    "requests": {"cpu": None, "memory": None, "gpu": None,"ephemeral_storage": None},
                    "limits": {"cpu": None, "memory": None, "gpu": None,"ephemeral_storage": None},
                },
                "expected_scheduler_resources": {
                    "requests": {},
                    "limits": {},
                },
                "expected_worker_resources": {
                    "requests": {},
                    "limits": {},
                },
            },
            {
                "default_function_pod_resources": {  # with defaults
                    "requests": {
                        "cpu": "25m",
                        "memory": "1M",
                    },
                    "limits": {"cpu": "2", "memory": "1G"},
                },
                "expected_scheduler_resources": {
                    "requests": {
                        "cpu": "25m",
                        "memory": "1M",
                    },
                    "limits": {"cpu": "2", "memory": "1G"},
                },
                "expected_worker_resources": {
                    "requests": {
                        "cpu": "25m",
                        "memory": "1M",
                    },
                    "limits": {"cpu": "2", "memory": "1G"},
                },
            },
        ]:
            mlrun.mlconf.default_function_pod_resources = test_case.get(
                "default_function_pod_resources"
            )

            runtime: mlrun.runtimes.DaskCluster = self._generate_runtime()
            expected_worker_resources = test_case.setdefault(
                "expected_worker_resources", {}
            )
            expected_scheduler_resources = test_case.setdefault(
                "expected_scheduler_resources", {}
            )

            expected_worker_requests = expected_worker_resources.get("requests")
            expected_worker_limits = expected_worker_resources.get("limits")
            expected_scheduler_requests = expected_scheduler_resources.get("requests")
            expected_scheduler_limits = expected_scheduler_resources.get("limits")

            _ = runtime.client
            self._assert_pods_resources(
                expected_worker_requests,
                expected_worker_limits,
                expected_scheduler_requests,
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

    def test_dask_with_default_security_context(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        _ = runtime.client
        self.kube_cluster_mock.assert_called_once()
        self.assert_security_context()

        default_security_context_dict = {
            "runAsUser": 1000,
            "runAsGroup": 3000,
        }
        default_security_context = self._generate_security_context(
            default_security_context_dict["runAsUser"],
            default_security_context_dict["runAsGroup"],
        )

        mlrun.mlconf.function.spec.security_context.default = base64.b64encode(
            json.dumps(default_security_context_dict).encode("utf-8")
        )
        runtime = self._generate_runtime()

        _ = runtime.client
        assert self.kube_cluster_mock.call_count == 2
        self.assert_security_context(default_security_context)

    def test_dask_with_security_context(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        other_security_context = self._generate_security_context(
            2000,
            2000,
        )

        # override security context
        runtime.with_security_context(other_security_context)
        _ = runtime.client
        self.assert_security_context(other_security_context)

    def test_deploy_dask_function_with_enriched_security_context(
        self, db: Session, client: TestClient, k8s_secrets_mock: K8sSecretsMock
    ):
        runtime = self._generate_runtime()
        user_unix_id = 1000
        auth_info = mlrun.api.schemas.AuthInfo(user_unix_id=user_unix_id)
        mlrun.mlconf.igz_version = "3.6"
        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            mlrun.api.schemas.function.SecurityContextEnrichmentModes.disabled.value
        )
        _ = mlrun.api.api.endpoints.functions._start_function(runtime, auth_info)
        pod = self._get_pod_creation_args()
        print(pod)
        self.assert_security_context()

        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            mlrun.api.schemas.function.SecurityContextEnrichmentModes.override.value
        )
        runtime = self._generate_runtime()
        _ = mlrun.api.api.endpoints.functions._start_function(runtime, auth_info)
        self.assert_security_context(
            self._generate_security_context(
                run_as_group=mlrun.mlconf.function.spec.security_context.enrichment_group_id,
                run_as_user=user_unix_id,
            )
        )
