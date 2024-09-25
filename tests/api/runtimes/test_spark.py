# Copyright 2023 Iguazio
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
import typing
import unittest

import deepdiff
import fastapi.testclient
import kubernetes
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.pod
import server.api.utils.singletons.k8s
import tests.api.runtimes.base
from mlrun.datastore import ParquetTarget
from mlrun.feature_store import RunConfig
from mlrun.feature_store.retrieval.job import _default_merger_handler


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

    def _assert_merged_node_selectors(
        self,
        body: dict,
        expected_job_node_selector: dict,
        expected_driver_node_selector: dict,
        expected_executor_node_selector: dict,
    ):
        if expected_job_node_selector:
            assert body["spec"]["nodeSelector"] == expected_job_node_selector
        if expected_driver_node_selector:
            assert (
                body["spec"]["driver"]["nodeSelector"] == expected_driver_node_selector
            )
        if expected_executor_node_selector:
            assert (
                body["spec"]["executor"]["nodeSelector"]
                == expected_executor_node_selector
            )

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
        expected_code: typing.Optional[str] = None,
    ):
        if assert_create_custom_object_called:
            server.api.utils.singletons.k8s.get_k8s_helper().crdapi.create_namespaced_custom_object.assert_called_once()

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

        if expected_code:
            body = self._get_custom_object_creation_body()
            code = None
            for envvar in body["spec"]["driver"]["env"]:
                if envvar["name"] == "MLRUN_EXEC_CODE":
                    code = envvar["value"]
                    break
            if code:
                code = base64.b64decode(code).decode("UTF-8")
            assert code == expected_code

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
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        mlrun.mlconf.httpdb.builder.docker_registry = "test_registry"
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        runtime.spec.image = None
        runtime.spec.use_default_image = True
        self.execute_function(runtime)
        self._assert_custom_object_creation_config()

    def test_run_without_runspec(self, db: sqlalchemy.orm.Session, k8s_secrets_mock):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_custom_object_creation_config()

    def test_run_with_default_resources(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )

        expected_executor_resources = {
            "requests": {"cpu": "1", "mem": "5g"},
            "limits": {"cpu": "2"},
        }
        expected_driver_resources = {
            "requests": {"cpu": "1", "mem": "2g"},
            "limits": {"cpu": "2"},
        }

        expected_cores = {
            "executor": 1,
            "driver": 1,
        }
        runtime.with_cores(expected_cores["executor"], expected_cores["driver"])

        self.execute_function(runtime)
        self._assert_custom_object_creation_config(
            expected_driver_resources=expected_driver_resources,
            expected_executor_resources=expected_executor_resources,
            expected_cores=expected_cores,
        )

    def test_run_with_limits_and_requests(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
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

    def test_run_with_conflicting_limits_and_requests(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )

        runtime.spec.service_account = "executorsa"
        runtime.with_executor_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="200m", gpus=1)

        runtime.with_driver_requests(cpu="2", mem="512m")
        runtime.with_driver_limits(cpu="3", gpus=1)

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            self.execute_function(runtime)

    def test_run_with_invalid_requests(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )
        with pytest.raises(ValueError):
            # Java notation applies to spark-operator memory requests
            runtime.with_driver_requests(mem="2Gi", cpu="3")

    def test_run_with_invalid_limits(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )
        with pytest.raises(ValueError):
            runtime.with_driver_limits(cpu="not a number", gpus=1)

    def test_run_with_limits_and_requests_patch_true(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
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
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
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
            "requests": {"mem": "1G", "cpu": "1"},
            "limits": {"cpu": "10"},
        }

        runtime.with_executor_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="3")

        # default patch = False
        runtime.with_executor_requests(mem="2G")
        runtime.with_executor_limits(cpu="5")
        expected_executor_resources = {
            "requests": {"mem": "2G", "cpu": "1"},
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

    @pytest.mark.parametrize(
        "project_node_selector,config_node_selector,function_node_selector,driver_node_selector,executor_node_selector",
        [
            # All parameters are empty
            ({}, {}, {}, None, None),
            # Only project node selector is defined
            ({"project-label": "project-val"}, {}, {}, None, None),
            # Only config node selector is defined
            ({}, {"config-label": "config-val"}, {}, None, None),
            # Only function node selector is defined
            ({}, {}, {"function-label": "function-val"}, None, None),
            # Only driver node selector is defined
            ({}, {}, {}, {"driver-label": "driver-val"}, None),
            # Only executor node selector is defined
            ({}, {}, {}, None, {"executor-label": "executor-val"}),
            # Project and function node selectors are defined
            (
                {"project-label": "project-val"},
                {},
                {"function-label": "function-val"},
                None,
                None,
            ),
            # Function selector is defined and overridden in driver
            (
                {},
                {},
                {"function-label": "function-val"},
                {},
                None,
            ),
            # Function selector is defined and overridden in executors
            (
                {},
                {},
                {"function-label": "function-val"},
                None,
                {},
            ),
            # Driver and executor node selectors are defined
            (
                {},
                {},
                {},
                {"driver-label": "driver-val"},
                {"executor-label": "executor-val"},
            ),
            # Project, config, and function node selectors are defined
            (
                {"project-label": "project-val"},
                {"config-label": "config-val"},
                {"function-label": "function-val"},
                None,
                None,
            ),
            # Project, driver, and executor node selectors are defined
            (
                {"project-label": "project-val"},
                {},
                {},
                {"driver-label": "driver-val"},
                {"executor-label": "executor-val"},
            ),
            # Config, driver, and executor node selectors are defined
            (
                {},
                {"config-label": "config-val"},
                {},
                {"driver-label": "driver-val"},
                {"executor-label": "executor-val"},
            ),
            # Function, driver, and executor node selectors are defined
            (
                {},
                {},
                {"function-label": "function-val"},
                {"driver-label": "driver-val"},
                {"executor-label": "executor-val"},
            ),
            # All node selectors are defined
            (
                {"project-label": "project-val"},
                {"config-label": "config-val"},
                {"function-label": "function-val"},
                {"driver-label": "driver-val"},
                {"executor-label": "executor-val"},
            ),
        ],
    )
    def test_with_node_selector(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock,
        project_node_selector,
        config_node_selector,
        function_node_selector,
        driver_node_selector,
        executor_node_selector,
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )

        run_db = mlrun.get_run_db()
        project = run_db.get_project(self.project)
        project.spec.default_function_node_selector = project_node_selector
        run_db.store_project(self.project, project)

        mlrun.mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(config_node_selector).encode("utf-8")
        )

        runtime.with_node_selection(node_selector=function_node_selector)
        runtime.with_executor_node_selection(node_selector=executor_node_selector)
        runtime.with_driver_node_selection(node_selector=driver_node_selector)

        self.execute_function(runtime)
        body = self._get_custom_object_creation_body()

        self._assert_merged_node_selectors(
            body,
            {},
            {
                **config_node_selector,
                **project_node_selector,
                **(
                    function_node_selector
                    if driver_node_selector is None
                    else driver_node_selector
                ),
            },
            {
                **config_node_selector,
                **project_node_selector,
                **(
                    function_node_selector
                    if executor_node_selector is None
                    else executor_node_selector
                ),
            },
        )

    def test_explicit_node_selector_for_function_applied_correctly(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )
        function_node_selector = {"function-label": "function-val"}
        # This test verifies that directly setting the node selector on the runtime object,
        # instead of using the `with_node_selection` method, correctly applies the function node selector
        # to the driver and executor. It also confirms that this direct setting does not affect the application spec.
        runtime.spec.node_selector = function_node_selector

        self.execute_function(runtime)
        body = self._get_custom_object_creation_body()
        self._assert_merged_node_selectors(
            body, {}, function_node_selector, function_node_selector
        )

    def test_with_node_selection_invalid_ns(self):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime(
            set_resources=False
        )
        function_node_selector = {"function-label": "function=val"}
        with pytest.warns(
            Warning,
            match="The node selector you’ve set does not meet the validation rules for the current Kubernetes version",
        ):
            runtime.with_node_selection(node_selector=function_node_selector)

    def test_run_with_host_path_volume(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
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

    def test_java_options(self, db: sqlalchemy.orm.Session, k8s_secrets_mock):
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
        k8s_secrets_mock,
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
        k8s_secrets_mock,
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
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        # no image pull secret
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_image_pull_secret()

        # default image pull secret
        mlrun.mlconf.function.spec.image_pull_secret.default = "my_secret"
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_image_pull_secret(
            mlrun.mlconf.function.spec.image_pull_secret.default,
        )

        # override default image pull secret
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        new_image_pull_secret = "my_new_secret"
        runtime.spec.image_pull_secret = new_image_pull_secret
        self.execute_function(runtime)
        self._assert_image_pull_secret(new_image_pull_secret)

    def test_get_offline_features(
        self,
        db: sqlalchemy.orm.Session,
        client: fastapi.testclient.TestClient,
        k8s_secrets_mock,
    ):
        # TODO - this test needs to be moved outside of the api runtimes tests and into the spark runtime sdk tests
        #   once moved, the `watch=False` can be removed
        import mlrun.feature_store as fstore

        fv = fstore.FeatureVector("my-vector", features=[])
        fv.save = unittest.mock.Mock()

        runtime = self._generate_runtime()
        # auto mount requires auth info but this test is supposed to run in the client
        # re-enable when test is moved
        runtime.spec.disable_auto_mount = True
        runtime.with_igz_spark = unittest.mock.Mock()

        self._reset_mocks()

        mlrun.mlconf.artifact_path = "v3io:///mypath"

        runtime.with_driver_limits(cpu="1")
        runtime.with_driver_requests(cpu="1", mem="1G")
        runtime.with_executor_limits(cpu="1")
        runtime.with_executor_requests(cpu="1", mem="1G")

        # remote-spark is not a merge engine but a runtime
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fstore.get_offline_features(
                fv,
                with_indexes=True,
                timestamp_for_filtering="timestamp",
                engine="remote-spark",
                run_config=RunConfig(local=False, function=runtime, watch=False),
                target=ParquetTarget(),
            )

        self.project = "default"
        self.project_default_function_node_selector = {}
        self._create_project(client)

        resp = fstore.get_offline_features(
            fv,
            with_indexes=True,
            timestamp_for_filtering="timestamp",
            engine="spark",
            # setting watch=False, because we don't want to wait for the job to complete when running in API
            run_config=RunConfig(local=False, function=runtime, watch=False),
            target=ParquetTarget(),
        )
        runspec = resp.run.spec.to_dict()
        expected_runspec = {
            "parameters": {
                "vector_uri": "store://feature-vectors/default/my-vector",
                "target": {
                    "name": "parquet",
                    "kind": "parquet",
                    "partitioned": True,
                    "max_events": 10000,
                    "flush_after_seconds": 900,
                },
                "entity_timestamp_column": None,
                "drop_columns": None,
                "with_indexes": True,
                "query": None,
                "order_by": None,
                "start_time": None,
                "end_time": None,
                "timestamp_for_filtering": "timestamp",
                "engine_args": None,
                "additional_filters": None,
            },
            "output_path": "v3io:///mypath",
            "function": "None/my-vector-merger@349f744e83e1a71d8b1faf4bbf3723dc0625daed",
            "handler": "merge_handler",
            "state_thresholds": mlrun.mlconf.function.spec.state_thresholds.default.to_dict(),
        }
        assert (
            deepdiff.DeepDiff(
                runspec,
                expected_runspec,
                # excluding function attribute as it contains hash of the object, excluding this path because any change
                # in the structure of the run will require to update the function hash
                exclude_paths=["root['function']"],
            )
            == {}
        )

        self.name = "my-vector-merger"

        expected_code = _default_merger_handler.replace(
            "{{{engine}}}", "SparkFeatureMerger"
        )

        self._assert_custom_object_creation_config(
            expected_driver_resources={
                "requests": {"cpu": "1", "mem": "1G"},
                "limits": {"cpu": "1"},
            },
            expected_executor_resources={
                "requests": {"cpu": "1", "mem": "1G"},
                "limits": {"cpu": "1"},
            },
            expected_code=expected_code,
        )

    def test_run_with_source_archive_pull_at_runtime(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="pull_at_runtime is not supported for spark runtime, use pull_at_runtime=False",
        ):
            runtime.with_source_archive(source="git://github.com/mock/repo")

        runtime.with_source_archive(
            source="git://github.com/mock/repo", pull_at_runtime=False
        )

    def test_run_with_load_source_on_run(
        self, db: sqlalchemy.orm.Session, k8s_secrets_mock
    ):
        # set default output path
        mlrun.mlconf.artifact_path = "v3io:///tmp"
        # generate runtime and set source code to load on run
        runtime: mlrun.runtimes.Spark3Runtime = self._generate_runtime()
        runtime.metadata.name = "test-spark-runtime"
        runtime.metadata.project = self.project
        runtime.spec.build.source = "git://github.com/mock/repo"
        runtime.spec.build.load_source_on_run = True
        # expect pre-condition error, not supported
        with pytest.raises(mlrun.errors.MLRunPreconditionFailedError) as exc:
            runtime.run(auth_info=mlrun.common.schemas.AuthInfo())

        assert (
            str(exc.value) == "Sparkjob does not support loading source code on run, "
            "use func.with_source_archive(pull_at_runtime=False)"
        )
