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
import os
import typing
import unittest

import deepdiff
import fastapi.testclient
import kubernetes
import pytest
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s
import mlrun.errors
import mlrun.runtimes.pod
import tests.api.runtimes.base
from mlrun.datastore import ParquetTarget
from mlrun.feature_store import RunConfig
pytestmark = pytest.mark.usefixtures("get_logs_api_mock")


class TestSpark3Runtime(tests.api.runtimes.base.TestRuntimeBase):
    def custom_setup_after_fixtures(self):
        self._mock_create_namespaced_custom_object()

    def custom_setup(self):
        self.image_name = "mlrun/mlrun:latest"

    def _generate_runtime(
        self, set_resources: bool = True
    ) -> mlrun.runtimes.Spark3Runtime:
        runtime = mlrun.runtimes.Spark3Runtime()
        runtime.spec.image = self.image_name
        if set_resources:
            runtime.with_executor_requests(cpu=1, mem="512m")
            runtime.with_driver_requests(cpu=1, mem="512m")
        return runtime

    def _assert_java_options(
        self,
        body: dict,
        expected_driver_java_options: str,
        expected_executor_java_options: str,
    ):
        if expected_driver_java_options:
            assert body["spec"]["driver"]["javaOptions"] == expected_driver_java_options
        else:
            assert "javaOptions" not in body["spec"]["driver"]
        if expected_executor_java_options:
            assert (
                body["spec"]["executor"]["javaOptions"]
                == expected_executor_java_options
            )
        else:
            assert "javaOptions" not in body["spec"]["executor"]

    @staticmethod
    def _assert_cores(body: dict, expected_cores: dict):
        for resource in ["executor", "driver"]:
            if expected_cores.get(resource):
                assert body[resource]["cores"] == expected_cores[resource]

    def _assert_custom_object_creation_config(
        self,
        expected_runtime_class_name="spark",
        assert_create_custom_object_called=True,
        expected_volumes: typing.Optional[list] = None,
        expected_driver_volume_mounts: typing.Optional[list] = None,
        expected_executor_volume_mounts: typing.Optional[list] = None,
        expected_driver_java_options=None,
        expected_executor_java_options=None,
        expected_driver_resources: dict = None,
        expected_executor_resources: dict = None,
        expected_cores: dict = None,
    ):
        if assert_create_custom_object_called:
            mlrun.api.utils.singletons.k8s.get_k8s().crdapi.create_namespaced_custom_object.assert_called_once()

        assert self._get_create_custom_object_namespace_arg() == self.namespace

        body = self._get_custom_object_creation_body()
        self._assert_labels(body["metadata"]["labels"], expected_runtime_class_name)
        self._assert_volume_and_mounts(
            body,
            expected_volumes,
            expected_driver_volume_mounts,
            expected_executor_volume_mounts,
        )

        self._assert_java_options(
            body, expected_driver_java_options, expected_executor_java_options
        )

        if expected_driver_resources:
            self._assert_resources(body["spec"]["driver"], expected_driver_resources)
        if expected_executor_resources:
            self._assert_resources(
                body["spec"]["executor"], expected_executor_resources
            )

        if expected_cores:
            self._assert_cores(body["spec"], expected_cores)

    def _assert_volume_and_mounts(
        self,
        body: dict,
        expected_volumes: typing.Optional[list] = None,
        expected_driver_volume_mounts: typing.Optional[list] = None,
        expected_executor_volume_mounts: typing.Optional[list] = None,
    ):
        if expected_volumes is not None:
            sanitized_volumes = self._sanitize_list_for_serialization(expected_volumes)
            assert (
                deepdiff.DeepDiff(
                    body["spec"]["volumes"],
                    sanitized_volumes,
                    ignore_order=True,
                    report_repetition=True,
                )
                == {}
            )
        if expected_driver_volume_mounts is not None:
            sanitized_driver_volume_mounts = self._sanitize_list_for_serialization(
                expected_driver_volume_mounts
            )
            assert (
                deepdiff.DeepDiff(
                    body["spec"]["driver"]["volumeMounts"],
                    sanitized_driver_volume_mounts,
                    ignore_order=True,
                    report_repetition=True,
                )
                == {}
            )
        if expected_executor_volume_mounts is not None:
            sanitized_executor_volume_mounts = self._sanitize_list_for_serialization(
                expected_executor_volume_mounts
            )
            assert (
                deepdiff.DeepDiff(
                    body["spec"]["executor"]["volumeMounts"],
                    sanitized_executor_volume_mounts,
                    ignore_order=True,
                    report_repetition=True,
                )
                == {}
            )

    def _assert_resources(self, actual_resources, expected_values):
        self._assert_limits(actual_resources, expected_values["limits"])
        self._assert_requests(actual_resources, expected_values["requests"])

    @staticmethod
    def _assert_requests(actual: dict, expected: dict):
        assert actual.get("coreRequest", None) == expected.get("cpu", None)
        assert actual.get("memory", None) == expected.get("mem", None)
        assert actual.get("serviceAccount", None) == expected.get(
            "serviceAccount", "sparkapp"
        )

    @staticmethod
    def _assert_limits(actual: dict, expected: dict):
        assert actual.get("coreLimit", None) == expected.get("cpu", None)
        assert actual.get("gpu", {}).get("name", None) == expected.get("gpu_type", None)
        assert actual.get("gpu", {}).get("quantity", None) == expected.get("gpus", None)

    def _assert_security_context(
        self,
        expected_driver_security_context=None,
        expected_executor_security_context=None,
    ):

        body = self._get_custom_object_creation_body()

        if expected_driver_security_context:
            assert (
                body["spec"]["driver"].get("securityContext")
                == expected_driver_security_context
            )
        else:
            assert body["spec"]["driver"].get("securityContext") is None

        if expected_executor_security_context:
            assert (
                body["spec"]["executor"].get("securityContext")
                == expected_executor_security_context
            )
        else:
            assert body["spec"]["executor"].get("securityContext") is None

    def _assert_image_pull_secret(
        self,
        expected_image_pull_secret=None,
    ):

        body = self._get_custom_object_creation_body()
        if expected_image_pull_secret:
            assert body["spec"].get("imagePullSecrets") == mlrun.utils.helpers.as_list(
                expected_image_pull_secret
            )
        else:
            assert body["spec"].get("imagePullSecrets") is None

    def _sanitize_list_for_serialization(self, list_: list):
        kubernetes_api_client = kubernetes.client.ApiClient()
        return list(map(kubernetes_api_client.sanitize_for_serialization, list_))

    def test_deploy_default_image_without_limits(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        mlrun.config.config.httpdb.builder.docker_registry = "test_registry"
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        runtime.spec.image = None
        runtime.spec.use_default_image = True
        self.execute_function(runtime)
        self._assert_custom_object_creation_config()

    def test_run_without_runspec(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_custom_object_creation_config()

    def test_run_without_required_resources(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
            self.execute_function(runtime)
        assert exc.value.args[0] == "Sparkjob must contain executor requests"

    def test_run_with_limits_and_requests(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )

        expected_executor_resources = {
            "requests": {"cpu": "1", "mem": "1G", "serviceAccount": "executorsa"},
            "limits": {"cpu": "2", "gpu_type": "nvidia.com/gpu", "gpus": 1},
        }
        expected_driver_resources = {
            "requests": {"cpu": "2", "mem": "512m"},
            "limits": {"cpu": "3", "gpu_type": "nvidia.com/gpu", "gpus": 1},
        }

        runtime.spec.service_account = "executorsa"
        runtime.with_executor_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="2", gpus=1)

        runtime.with_driver_requests(cpu="2", mem="512m")
        runtime.with_driver_limits(cpu="3", gpus=1)

        expected_cores = {
            "executor": 8,
            "driver": 2,
        }
        runtime.with_cores(expected_cores["executor"], expected_cores["driver"])

        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_driver_resources=expected_driver_resources,
            expected_executor_resources=expected_executor_resources,
            expected_cores=expected_cores,
        )

    def test_run_with_limits_and_requests_patch_true(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )

        runtime.with_executor_limits(cpu="3")
        runtime.with_executor_requests(cpu="1", mem="1G")

        runtime.with_executor_limits(gpus=1, patch=True)
        expected_executor_resources = {
            "requests": {"cpu": "1", "mem": "1G"},
            "limits": {"cpu": "3", "gpu_type": "nvidia.com/gpu", "gpus": 1},
        }

        runtime.with_driver_requests(cpu="2")
        runtime.with_driver_limits(cpu="3", gpus=1)
        # patch = True
        runtime.with_driver_requests(mem="512m", patch=True)
        expected_driver_resources = {
            "requests": {"cpu": "2", "mem": "512m"},
            "limits": {"cpu": "3", "gpu_type": "nvidia.com/gpu", "gpus": 1},
        }

        expected_cores = {
            "executor": 8,
            "driver": 2,
        }
        runtime.with_cores(expected_cores["executor"], expected_cores["driver"])

        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_driver_resources=expected_driver_resources,
            expected_executor_resources=expected_executor_resources,
            expected_cores=expected_cores,
        )

    def test_run_with_limits_and_requests_patch_false(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )
        runtime.with_driver_requests(cpu="2")
        runtime.with_driver_limits(cpu="3", gpus=1)

        # default patch = False
        runtime.with_driver_requests(mem="1G")
        runtime.with_driver_limits(cpu="10")
        expected_driver_resources = {
            "requests": {"mem": "1G"},
            "limits": {"cpu": "10"},
        }

        runtime.with_executor_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="3")

        # default patch = False
        runtime.with_executor_requests(mem="2G")
        runtime.with_executor_limits(cpu="5")
        expected_executor_resources = {
            "requests": {"mem": "2G"},
            "limits": {"cpu": "5"},
        }
        expected_cores = {
            "executor": 8,
            "driver": 2,
        }
        runtime.with_cores(expected_cores["executor"], expected_cores["driver"])

        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_driver_resources=expected_driver_resources,
            expected_executor_resources=expected_executor_resources,
            expected_cores=expected_cores,
        )

    def test_run_with_host_path_volume(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        shared_volume = kubernetes.client.V1Volume(
            name="shared-volume",
            host_path=kubernetes.client.V1HostPathVolumeSource(
                path="/shared-volume-host-path", type=""
            ),
        )
        shared_volume_driver_volume_mount = kubernetes.client.V1VolumeMount(
            mount_path="/shared-volume-driver-mount-path", name=shared_volume.name
        )
        shared_volume_executor_volume_mount = kubernetes.client.V1VolumeMount(
            mount_path="/shared-volume-executor-mount-path", name=shared_volume.name
        )
        driver_volume = kubernetes.client.V1Volume(
            name="driver-volume",
            host_path=kubernetes.client.V1HostPathVolumeSource(
                path="/driver-volume-host-path", type=""
            ),
        )
        driver_volume_volume_mount = kubernetes.client.V1VolumeMount(
            mount_path="/driver-mount-path", name=driver_volume.name
        )
        executor_volume = kubernetes.client.V1Volume(
            name="executor-volume",
            host_path=kubernetes.client.V1HostPathVolumeSource(
                path="/executor-volume-host-path", type=""
            ),
        )
        executor_volume_volume_mount = kubernetes.client.V1VolumeMount(
            mount_path="/executor-mount-path", name=executor_volume.name
        )
        runtime.with_driver_host_path_volume(
            shared_volume.host_path.path,
            shared_volume_driver_volume_mount.mount_path,
            volume_name=shared_volume.name,
        )
        runtime.with_executor_host_path_volume(
            shared_volume.host_path.path,
            shared_volume_executor_volume_mount.mount_path,
            volume_name=shared_volume.name,
        )
        runtime.with_driver_host_path_volume(
            driver_volume.host_path.path,
            driver_volume_volume_mount.mount_path,
            volume_name=driver_volume.name,
        )
        runtime.with_executor_host_path_volume(
            executor_volume.host_path.path,
            executor_volume_volume_mount.mount_path,
            volume_name=executor_volume.name,
        )
        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_volumes=[shared_volume, driver_volume, executor_volume],
            expected_driver_volume_mounts=[
                shared_volume_driver_volume_mount,
                driver_volume_volume_mount,
            ],
            expected_executor_volume_mounts=[
                shared_volume_executor_volume_mount,
                executor_volume_volume_mount,
            ],
        )

    def test_java_options(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime = self._generate_runtime()
        driver_java_options = "-Dmyproperty=somevalue"
        runtime.spec.driver_java_options = driver_java_options
        executor_java_options = "-Dmyotherproperty=someothervalue"
        runtime.spec.executor_java_options = executor_java_options
        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_driver_java_options=driver_java_options,
            expected_executor_java_options=executor_java_options,
        )

    @pytest.mark.parametrize(
        "executor_cores, driver_cores, expect_failure",
        [
            (4, None, False),
            (3, 3, False),
            (None, 2, False),
            (None, None, False),
            (0.5, None, True),
            (None, -1, True),
        ],
    )
    def test_cores(
        self,
        executor_cores,
        driver_cores,
        expect_failure,
        db: sqlalchemy.orm.Session,
        client: fastapi.testclient.TestClient,
    ):
        runtime = self._generate_runtime()
        if expect_failure:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                runtime.with_cores(
                    executor_cores=executor_cores, driver_cores=driver_cores
                )
            return
        else:
            runtime.with_cores(executor_cores=executor_cores, driver_cores=driver_cores)

        # By default, if not specified otherwise, the cores are set to 1
        expected_cores = {"executor": executor_cores or 1, "driver": driver_cores or 1}

        self.execute_function(runtime)
        self._assert_custom_object_creation_config(expected_cores=expected_cores)

    @pytest.mark.parametrize(
        ["mount_v3io_to_executor", "with_igz_spark_twice"],
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_with_igz_spark_volume_mounts(
        self,
        mount_v3io_to_executor,
        with_igz_spark_twice,
        db: sqlalchemy.orm.Session,
        client: fastapi.testclient.TestClient,
    ):
        runtime = self._generate_runtime()

        orig = os.getenv("V3IO_USERNAME")
        os.environ["V3IO_USERNAME"] = "me"
        try:
            runtime.with_executor_host_path_volume(
                host_path="/tmp",
                mount_path="/before",
                volume_name="path-volume-before",
            )
            runtime.with_igz_spark(mount_v3io_to_executor=mount_v3io_to_executor)
            if with_igz_spark_twice:
                runtime.with_igz_spark(mount_v3io_to_executor=mount_v3io_to_executor)
            runtime.with_executor_host_path_volume(
                host_path="/tmp",
                mount_path="/after",
                volume_name="path-volume-after",
            )
        finally:
            if orig:
                os.environ["V3IO_USERNAME"] = orig
            else:
                os.unsetenv("V3IO_USERNAME")

        self.execute_function(runtime)
        user_added_executor_volume_mounts = [
            kubernetes.client.V1VolumeMount(
                mount_path="/before", name="path-volume-before"
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/after", name="path-volume-after"
            ),
        ]
        common_volume_mounts = [
            kubernetes.client.V1VolumeMount(mount_path="/dev/shm", name="shm"),
            kubernetes.client.V1VolumeMount(
                mount_path="/var/run/iguazio/dayman", name="v3iod-comm"
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/var/run/iguazio/daemon_health", name="daemon-health"
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/etc/config/v3io", name="v3io-config"
            ),
        ]
        v3io_mounts = [
            kubernetes.client.V1VolumeMount(
                mount_path="/v3io", name="v3io", sub_path=""
            ),
            kubernetes.client.V1VolumeMount(
                mount_path="/User", name="v3io", sub_path="users/me"
            ),
        ]
        expected_driver_mounts = common_volume_mounts + v3io_mounts
        expected_executor_mounts = (
            common_volume_mounts + user_added_executor_volume_mounts
        )
        if mount_v3io_to_executor:
            expected_executor_mounts += v3io_mounts
        self._assert_custom_object_creation_config(
            expected_driver_volume_mounts=expected_driver_mounts,
            expected_executor_volume_mounts=expected_executor_mounts,
        )

    def test_deploy_with_image_pull_secret(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):

        # no image pull secret
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_image_pull_secret()

        # default image pull secret
        mlrun.config.config.function.spec.image_pull_secret.default = "my_secret"
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_image_pull_secret(
            mlrun.config.config.function.spec.image_pull_secret.default,
        )

        # override default image pull secret
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        new_image_pull_secret = "my_new_secret"
        runtime.spec.image_pull_secret = new_image_pull_secret
        self.execute_function(runtime)
        self._assert_image_pull_secret(new_image_pull_secret)

    def test_get_offline_features(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        import mlrun.feature_store as fstore

        fv = fstore.FeatureVector("my-vector", features=[])
        fv.save = unittest.mock.Mock()

        self._reset_mocks()
        runtime = self._generate_runtime()
        runtime.spec.output_path = "v3io:///mypath"

        runtime.with_driver_limits(cpu="1")
        runtime.with_driver_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="1")
        runtime.with_executor_requests(cpu="1", mem="1G")

        resp = fstore.get_offline_features(
            fv,
            with_indexes=True,
            entity_timestamp_column="timestamp",
            engine="spark",
            run_config=RunConfig(local=False, function=runtime),
            target=ParquetTarget(),
        )
        runspec = resp.run.spec.to_dict()
        assert runspec == {
            "parameters": {
                "vector_uri": "store://feature-vectors/default/my-vector",
                "target": {
                    "name": "parquet",
                    "kind": "parquet",
                    "partitioned": True,
                    "max_events": 10000,
                    "flush_after_seconds": 900,
                },
                "timestamp_column": "timestamp",
                "drop_columns": None,
                "with_indexes": True,
                "query": None,
                "engine_args": None,
            },
            "outputs": [],
            "output_path": "v3io:///mypath",
            "function": "None/my-vector_merger@0f4fef1da6f72c229b33fefbff0e5b58d87263c7",
            "secret_sources": [],
            "data_stores": [],
            "handler": "merge_handler",
        }
