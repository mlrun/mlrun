import typing

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

    def _assert_volume_and_mounts(
        self,
        body: dict,
        expected_volumes: typing.Optional[list] = None,
        expected_driver_volume_mounts: typing.Optional[list] = None,
        expected_executor_volume_mounts: typing.Optional[list] = None,
    ):
        if expected_volumes:
            sanitized_volumes = self._sanitize_list_for_serialization(expected_volumes)
            assert (
                deepdiff.DeepDiff(
                    body["spec"]["volumes"], sanitized_volumes, ignore_order=True
                )
                == {}
            )
        if expected_driver_volume_mounts:
            sanitized_driver_volume_mounts = self._sanitize_list_for_serialization(
                expected_driver_volume_mounts
            )
            assert (
                deepdiff.DeepDiff(
                    body["spec"]["driver"]["volumeMounts"],
                    sanitized_driver_volume_mounts,
                    ignore_order=True,
                )
                == {}
            )
        if expected_executor_volume_mounts:
            sanitized_executor_volume_mounts = self._sanitize_list_for_serialization(
                expected_executor_volume_mounts
            )
            assert (
                deepdiff.DeepDiff(
                    body["spec"]["executor"]["volumeMounts"],
                    sanitized_executor_volume_mounts,
                    ignore_order=True,
                )
                == {}
            )

    def _assert_resources(self, actual_resources, expected_values):
        self._assert_limits(actual_resources, expected_values["limits"])
        self._assert_requests(actual_resources, expected_values["requests"])

    @staticmethod
    def _assert_requests(actual, expected):
        assert actual["coreRequest"] == expected["cpu"]
        assert actual["memory"] == expected["mem"]

    @staticmethod
    def _assert_limits(actual, expected):
        assert actual["coreLimit"] == expected["cpu"]
        assert actual["gpu"]["name"] == expected["gpu_type"]
        assert actual["gpu"]["quantity"] == expected["gpus"]

    def _sanitize_list_for_serialization(self, list_: list):
        kubernetes_api_client = kubernetes.client.ApiClient()
        return list(map(kubernetes_api_client.sanitize_for_serialization, list_))

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

        expected_executor = {
            "requests": {"cpu": "1", "mem": "1G"},
            "limits": {"cpu": "2", "gpu_type": "nvidia.com/gpu", "gpus": 1},
        }
        expected_driver = {
            "requests": {"cpu": "2", "mem": "512m"},
            "limits": {"cpu": "3", "gpu_type": "nvidia.com/gpu", "gpus": 1},
        }

        runtime.with_executor_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="2", gpus=1)

        runtime.with_driver_requests(cpu="2", mem="512m")
        runtime.with_driver_limits(cpu="3", gpus=1)

        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_driver_resources=expected_driver,
            expected_executor_resources=expected_executor,
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
