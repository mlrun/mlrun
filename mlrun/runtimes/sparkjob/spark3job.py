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
import typing

import kubernetes.client

import mlrun.common.schemas.function
import mlrun.errors
import mlrun.k8s_utils
import mlrun.runtimes.pod
from mlrun.config import config
from mlrun.runtimes.mounts import mount_v3io, mount_v3iod

from ...execution import MLClientCtx
from ...model import RunObject
from ...utils import update_in, verify_field_regex
from ..kubejob import KubejobRuntime
from ..pod import KubeResourceSpec
from ..utils import (
    generate_resources,
    get_gpu_from_resource_requirement,
    get_item_name,
    verify_limits,
    verify_requests,
)


class Spark3JobSpec(KubeResourceSpec):
    _jvm_memory_resource_notation = r"^[0-9]+[KkMmGg]$"

    # https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/55732a6a392cbe1d6546c7ec6823193ab055d2fa/pkg/apis/sparkoperator.k8s.io/v1beta2/types.go#L181
    _dict_fields = KubeResourceSpec._dict_fields + [
        "driver_resources",
        "executor_resources",
        "job_type",
        "python_version",
        "spark_version",
        "restart_policy",
        "deps",
        "main_class",
        "spark_conf",
        "hadoop_conf",
        "use_default_image",
        "monitoring",
        "driver_node_selector",
        "executor_node_selector",
        "dynamic_allocation",
        "driver_tolerations",
        "executor_tolerations",
        "driver_affinity",
        "executor_affinity",
        "driver_preemption_mode",
        "executor_preemption_mode",
        "driver_volume_mounts",
        "executor_volume_mounts",
        "driver_java_options",
        "executor_java_options",
        "driver_cores",
        "executor_cores",
    ]
    _default_fields_to_strip = KubeResourceSpec._default_fields_to_strip + [
        "driver_node_selector",
        "executor_node_selector",
        "driver_tolerations",
        "executor_tolerations",
        "driver_affinity",
        "executor_affinity",
        "driver_volume_mounts",
        "executor_volume_mounts",
        "driver_cores",
        "executor_cores",
    ]

    __k8s_fields_to_serialize = [
        "driver_volume_mounts",
        "executor_volume_mounts",
        "driver_node_selector",
        "executor_node_selector",
        "executor_affinity",
        "executor_tolerations",
        "driver_affinity",
        "driver_tolerations",
    ]
    _k8s_fields_to_serialize = (
        KubeResourceSpec._k8s_fields_to_serialize + __k8s_fields_to_serialize
    )
    _fields_to_serialize = (
        KubeResourceSpec._fields_to_serialize + __k8s_fields_to_serialize
    )
    _fields_to_skip_validation = KubeResourceSpec._fields_to_skip_validation + [
        # TODO: affinity, tolerations and node_selector are skipped due to preemption mode transitions.
        #  Preemption mode 'none' depends on the previous mode while the default mode may enrich these values.
        #  When we allow 'None' values for these attributes we get their true values and they will undo the default
        #  enrichment when creating the runtime from dict.
        #  The enrichment should move to the server side and then this can be removed.
        "driver_node_selector",
        "executor_node_selector",
        "executor_affinity",
        "executor_tolerations",
        "driver_affinity",
        "driver_tolerations",
    ]

    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        volumes=None,
        volume_mounts=None,
        env=None,
        resources=None,
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        image_pull_secret=None,
        driver_resources=None,
        executor_resources=None,
        job_type=None,
        python_version=None,
        spark_version=None,
        restart_policy=None,
        deps=None,
        main_class=None,
        default_handler=None,
        entry_points=None,
        description=None,
        workdir=None,
        build=None,
        spark_conf=None,
        hadoop_conf=None,
        node_selector=None,
        driver_node_selector=None,
        executor_node_selector=None,
        use_default_image=False,
        priority_class_name=None,
        dynamic_allocation=None,
        monitoring=None,
        disable_auto_mount=False,
        pythonpath=None,
        node_name=None,
        affinity=None,
        tolerations=None,
        driver_tolerations=None,
        executor_tolerations=None,
        executor_affinity=None,
        driver_affinity=None,
        preemption_mode=None,
        executor_preemption_mode=None,
        driver_preemption_mode=None,
        driver_volume_mounts=None,
        executor_volume_mounts=None,
        driver_java_options=None,
        executor_java_options=None,
        driver_cores=None,
        executor_cores=None,
        security_context=None,
        clone_target_dir=None,
        state_thresholds=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            replicas=replicas,
            image_pull_policy=image_pull_policy,
            service_account=service_account,
            image_pull_secret=image_pull_secret,
            default_handler=default_handler,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            build=build,
            node_selector=node_selector,
            priority_class_name=priority_class_name,
            disable_auto_mount=disable_auto_mount,
            pythonpath=pythonpath,
            node_name=node_name,
            affinity=affinity,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
            clone_target_dir=clone_target_dir,
            state_thresholds=state_thresholds,
        )

        self.driver_resources = driver_resources or {}
        self.executor_resources = executor_resources or {}
        self.spark_conf = spark_conf or {}
        self.hadoop_conf = hadoop_conf or {}
        self.job_type = job_type
        self.python_version = python_version
        self.spark_version = spark_version
        self.restart_policy = restart_policy or {}
        self.deps = deps or {}
        self.main_class = main_class
        self.use_default_image = use_default_image
        self.dynamic_allocation = dynamic_allocation or {}
        self.driver_node_selector = driver_node_selector
        self.executor_node_selector = executor_node_selector
        self.monitoring = monitoring or {}
        self.driver_tolerations = driver_tolerations
        self.executor_tolerations = executor_tolerations
        self.executor_affinity = executor_affinity
        self.driver_affinity = driver_affinity
        self.executor_preemption_mode = executor_preemption_mode
        self.driver_preemption_mode = driver_preemption_mode
        self._driver_volume_mounts = {}
        self.driver_volume_mounts = driver_volume_mounts or {}
        self._executor_volume_mounts = {}
        self.executor_volume_mounts = executor_volume_mounts or {}
        self.driver_java_options = driver_java_options
        self.executor_java_options = executor_java_options
        self.driver_cores = driver_cores
        self.executor_cores = executor_cores

    @property
    def executor_tolerations(self) -> list[kubernetes.client.V1Toleration]:
        return self._executor_tolerations

    @executor_tolerations.setter
    def executor_tolerations(self, executor_tolerations):
        self._executor_tolerations = (
            mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                "executor_tolerations", executor_tolerations
            )
        )

    @property
    def driver_tolerations(self) -> list[kubernetes.client.V1Toleration]:
        return self._driver_tolerations

    @driver_tolerations.setter
    def driver_tolerations(self, driver_tolerations):
        self._driver_tolerations = (
            mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                "driver_tolerations", driver_tolerations
            )
        )

    @property
    def executor_affinity(self) -> kubernetes.client.V1Affinity:
        return self._executor_affinity

    @executor_affinity.setter
    def executor_affinity(self, affinity):
        self._executor_affinity = (
            mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                "executor_affinity", affinity
            )
        )

    @property
    def driver_affinity(self) -> kubernetes.client.V1Affinity:
        return self._driver_affinity

    @driver_affinity.setter
    def driver_affinity(self, affinity):
        self._driver_affinity = (
            mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                "executor_affinity", affinity
            )
        )

    @property
    def driver_preemption_mode(self) -> str:
        return self._driver_preemption_mode

    @driver_preemption_mode.setter
    def driver_preemption_mode(self, mode):
        self._driver_preemption_mode = (
            mode or mlrun.mlconf.function_defaults.preemption_mode
        )
        self.enrich_function_preemption_spec(
            preemption_mode_field_name="driver_preemption_mode",
            tolerations_field_name="driver_tolerations",
            affinity_field_name="driver_affinity",
            node_selector_field_name="driver_node_selector",
        )

    @property
    def executor_preemption_mode(self) -> str:
        return self._executor_preemption_mode

    @executor_preemption_mode.setter
    def executor_preemption_mode(self, mode):
        self._executor_preemption_mode = (
            mode or mlrun.mlconf.function_defaults.preemption_mode
        )
        self.enrich_function_preemption_spec(
            preemption_mode_field_name="executor_preemption_mode",
            tolerations_field_name="executor_tolerations",
            affinity_field_name="executor_affinity",
            node_selector_field_name="executor_node_selector",
        )

    @property
    def driver_volume_mounts(self) -> list:
        return list(self._driver_volume_mounts.values())

    @driver_volume_mounts.setter
    def driver_volume_mounts(self, volume_mounts):
        self._driver_volume_mounts = {}
        if volume_mounts:
            for volume_mount in volume_mounts:
                self._set_volume_mount(
                    volume_mount, volume_mounts_field_name="_driver_volume_mounts"
                )

    @property
    def executor_volume_mounts(self) -> list:
        return list(self._executor_volume_mounts.values())

    @executor_volume_mounts.setter
    def executor_volume_mounts(self, volume_mounts):
        self._executor_volume_mounts = {}
        if volume_mounts:
            for volume_mount in volume_mounts:
                self._set_volume_mount(
                    volume_mount, volume_mounts_field_name="_executor_volume_mounts"
                )

    def _verify_jvm_memory_string(
        self, resources_field_name: str, memory: typing.Optional[str]
    ):
        if memory:
            verify_field_regex(
                f"function.spec.{resources_field_name}.requests.memory",
                memory,
                [self._jvm_memory_resource_notation],
            )

    def enrich_resources_with_default_pod_resources(
        self, resources_field_name: str, resources: dict
    ):
        if resources_field_name == "driver_resources":
            role = "driver"
        elif resources_field_name == "executor_resources":
            role = "executor"
        else:
            return {}
        resources_types = ["cpu", "memory"]
        resource_requirements = ["requests", "limits"]
        default_resources = mlrun.mlconf.default_spark_resources.to_dict()[role]

        if resources:
            for resource_requirement in resource_requirements:
                for resource_type in resources_types:
                    if (
                        resources.setdefault(resource_requirement, {}).setdefault(
                            resource_type
                        )
                        is None
                    ):
                        resources[resource_requirement][resource_type] = (
                            default_resources[resource_requirement][resource_type]
                        )
        else:
            resources = default_resources

        # Spark operator uses JVM notation for memory, so we must verify it separately
        verify_requests(
            resources_field_name,
            cpu=resources["requests"]["cpu"],
        )
        self._verify_jvm_memory_string(
            resources_field_name, resources["requests"]["memory"]
        )
        resources["requests"] = generate_resources(
            mem=resources["requests"]["memory"], cpu=resources["requests"]["cpu"]
        )
        gpu_type, gpu_value = get_gpu_from_resource_requirement(resources["limits"])
        verify_limits(
            resources_field_name,
            cpu=resources["limits"]["cpu"],
            gpus=gpu_value,
            gpu_type=gpu_type,
        )
        resources["limits"] = generate_resources(
            cpu=resources["limits"]["cpu"],
            gpus=gpu_value,
            gpu_type=gpu_type,
        )
        if not resources["requests"] and not resources["limits"]:
            return {}
        return resources

    def _verify_and_set_requests(
        self,
        resources_field_name,
        mem: typing.Optional[str] = None,
        cpu: typing.Optional[str] = None,
        patch: bool = False,
    ):
        # Spark operator uses JVM notation for memory, so we must verify it separately
        verify_requests(resources_field_name, cpu=cpu)
        self._verify_jvm_memory_string(resources_field_name, mem)
        resources = generate_resources(mem=mem, cpu=cpu)

        if not patch:
            update_in(
                getattr(self, resources_field_name),
                "requests",
                resources,
            )
        else:
            for resource, resource_value in resources.items():
                update_in(
                    getattr(self, resources_field_name),
                    f"requests.{resource}",
                    resource_value,
                )

    @property
    def driver_resources(self) -> dict:
        return self._driver_resources

    @driver_resources.setter
    def driver_resources(self, resources):
        self._driver_resources = self.enrich_resources_with_default_pod_resources(
            "driver_resources", resources
        )

    @property
    def executor_resources(self) -> dict:
        return self._executor_resources

    @executor_resources.setter
    def executor_resources(self, resources):
        self._executor_resources = self.enrich_resources_with_default_pod_resources(
            "executor_resources", resources
        )


class Spark3Runtime(KubejobRuntime):
    group = "sparkoperator.k8s.io"
    version = "v1beta2"
    apiVersion = group + "/" + version  # noqa: N815
    kind = "spark"
    plural = "sparkapplications"

    # the dot will make the api prefix the configured registry to the image name
    default_mlrun_image = ".spark-job-default-image"
    gpu_suffix = "-cuda"
    code_script = "spark-function-code.py"
    code_path = "/etc/config/mlrun"

    @property
    def spec(self) -> Spark3JobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", Spark3JobSpec)

    def _get_igz_deps(self):
        return {
            "jars": [
                "local:///spark/v3io-libs/v3io-hcfs_2.12.jar",
                "local:///spark/v3io-libs/v3io-spark3-streaming_2.12.jar",
                "local:///spark/v3io-libs/v3io-spark3-object-dataframe_2.12.jar",
                "local:///igz/java/libs/scala-library-2.12.14.jar",
                "local:///spark/jars/jmx_prometheus_javaagent-0.16.1.jar",
            ],
            "files": ["local:///igz/java/libs/v3io-pyspark.zip"],
        }

    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[dict[str, str]] = None,
        affinity: typing.Optional[kubernetes.client.V1Affinity] = None,
        tolerations: typing.Optional[list[kubernetes.client.V1Toleration]] = None,
    ):
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        if affinity:
            raise NotImplementedError(
                "Setting affinity is not supported for spark runtime"
            )
        if tolerations:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                "Tolerations can be set in spark runtime but not in with_node_selection. "
                "Instead, use with_driver_node_selection and with_executor_node_selection to set tolerations."
            )
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        mlrun.k8s_utils.validate_node_selectors(node_selector, raise_on_error=False)
        self.with_driver_node_selection(node_name, node_selector, affinity, tolerations)
        self.with_executor_node_selection(
            node_name, node_selector, affinity, tolerations
        )

    def with_driver_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[dict[str, str]] = None,
        affinity: typing.Optional[kubernetes.client.V1Affinity] = None,
        tolerations: typing.Optional[list[kubernetes.client.V1Toleration]] = None,
    ):
        """
        Enables control of which k8s node the spark executor will run on.

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels are eligible to be picked
        :param affinity:        Expands the types of constraints you can express - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
                                for details
        :param tolerations:     Tolerations are applied to pods, and allow (but do not require) the pods to schedule
                                onto nodes with matching taints - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration
                                for details
        """
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        if affinity is not None:
            self.spec.driver_affinity = affinity
        if node_selector is not None:
            mlrun.k8s_utils.validate_node_selectors(node_selector, raise_on_error=False)
            self.spec.driver_node_selector = node_selector
        if tolerations is not None:
            self.spec.driver_tolerations = tolerations

    def with_executor_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[dict[str, str]] = None,
        affinity: typing.Optional[kubernetes.client.V1Affinity] = None,
        tolerations: typing.Optional[list[kubernetes.client.V1Toleration]] = None,
    ):
        """
        Enables control of which k8s node the spark executor will run on.

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels are eligible to be picked
        :param affinity:        Expands the types of constraints you can express - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
                                for details
        :param tolerations:     Tolerations are applied to pods, and allow (but do not require) the pods to schedule
                                onto nodes with matching taints - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration
                                for details
        """
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        if affinity is not None:
            self.spec.executor_affinity = affinity
        if node_selector is not None:
            mlrun.k8s_utils.validate_node_selectors(node_selector, raise_on_error=False)
            self.spec.executor_node_selector = node_selector
        if tolerations is not None:
            self.spec.executor_tolerations = tolerations

    def with_preemption_mode(
        self, mode: typing.Union[mlrun.common.schemas.function.PreemptionModes, str]
    ):
        """
        Use with_driver_preemption_mode / with_executor_preemption_mode to setup preemption_mode for spark operator
        """
        raise mlrun.errors.MLRunInvalidArgumentTypeError(
            "with_preemption_mode is not supported, use with_driver_preemption_mode / with_executor_preemption_mode"
            " to set preemption mode for spark operator"
        )

    def with_driver_preemption_mode(
        self, mode: typing.Union[mlrun.common.schemas.function.PreemptionModes, str]
    ):
        """
        Preemption mode controls whether the spark driver can be scheduled on preemptible nodes.
        Tolerations, node selector, and affinity are populated on preemptible nodes corresponding to the function spec.

        The supported modes are:

        * **allow** - The function can be scheduled on preemptible nodes
        * **constrain** - The function can only run on preemptible nodes
        * **prevent** - The function cannot be scheduled on preemptible nodes
        * **none** - No preemptible configuration will be applied on the function

        The default preemption mode is configurable in mlrun.mlconf.function_defaults.preemption_mode.
        By default it's set to **prevent**

        :param mode: allow | constrain | prevent | none defined in :py:class:`~mlrun.common.schemas.PreemptionModes`
        """
        preemption_mode = mlrun.common.schemas.function.PreemptionModes(mode)
        self.spec.driver_preemption_mode = preemption_mode.value

    def with_executor_preemption_mode(
        self, mode: typing.Union[mlrun.common.schemas.function.PreemptionModes, str]
    ):
        """
        Preemption mode controls whether the spark executor can be scheduled on preemptible nodes.
        Tolerations, node selector, and affinity are populated on preemptible nodes corresponding to the function spec.

        The supported modes are:

        * **allow** - The function can be scheduled on preemptible nodes
        * **constrain** - The function can only run on preemptible nodes
        * **prevent** - The function cannot be scheduled on preemptible nodes
        * **none** - No preemptible configuration will be applied on the function

        The default preemption mode is configurable in mlrun.mlconf.function_defaults.preemption_mode,
        by default it's set to **prevent**

        :param mode: allow | constrain | prevent | none defined in :py:class:`~mlrun.common.schemas.PreemptionModes`
        """
        preemption_mode = mlrun.common.schemas.function.PreemptionModes(mode)
        self.spec.executor_preemption_mode = preemption_mode.value

    def with_security_context(
        self, security_context: kubernetes.client.V1SecurityContext
    ):
        """
        With security context is not supported for spark runtime.
        Driver / Executor processes run with uid / gid 1000 as long as security context is not defined.
        If in the future we want to support setting security context it will work only from spark version 3.2 onwards.
        """
        raise mlrun.errors.MLRunInvalidArgumentTypeError(
            "with_security_context is not supported with spark operator"
        )

    def with_driver_host_path_volume(
        self,
        host_path: str,
        mount_path: str,
        type: str = "",
        volume_name: str = "host-path-volume",
    ):
        """
        Add a host path volume and mounts it to the driver pod
        More info: https://kubernetes.io/docs/concepts/storage/volumes#hostpath

        :param host_path:   Path of the directory on the host. If the path is a symlink, it follows the link to the
                            real path
        :param mount_path:  Path within the container at which the volume should be mounted.  Must not contain ':'
        :param type:        Type for HostPath Volume Defaults to ""
        :param volume_name: Volume's name. Must be a DNS_LABEL and unique within the pod
        """
        self._with_host_path_volume(
            "_driver_volume_mounts", host_path, mount_path, type, volume_name
        )

    def with_executor_host_path_volume(
        self,
        host_path: str,
        mount_path: str,
        type: str = "",
        volume_name: str = "host-path-volume",
    ):
        """
        Add an host path volume and mount it to the executor pod/s
        More info: https://kubernetes.io/docs/concepts/storage/volumes#hostpath

        :param host_path:   Path of the directory on the host. If the path is a symlink, it follows the link to the
                            real path
        :param mount_path:  Path within the container at which the volume should be mounted.  Must not contain ':'
        :param type:        Type for HostPath Volume Defaults to ""
        :param volume_name: Volume's name. Must be a DNS_LABEL and unique within the pod
        """
        self._with_host_path_volume(
            "_executor_volume_mounts", host_path, mount_path, type, volume_name
        )

    def _with_host_path_volume(
        self,
        volume_mounts_field_name,
        host_path: str,
        mount_path: str,
        type_: str = "",
        volume_name: str = "host-path-volume",
    ):
        volume = kubernetes.client.V1Volume(
            name=volume_name,
            host_path=kubernetes.client.V1HostPathVolumeSource(
                path=host_path, type=type_
            ),
        )
        volume_mount = kubernetes.client.V1VolumeMount(
            mount_path=mount_path, name=volume_name
        )
        kubernetes_api_client = kubernetes.client.ApiClient()
        self.spec.update_vols_and_mounts(
            [kubernetes_api_client.sanitize_for_serialization(volume)],
            [kubernetes_api_client.sanitize_for_serialization(volume_mount)],
            volume_mounts_field_name,
        )

    def with_dynamic_allocation(
        self, min_executors=None, max_executors=None, initial_executors=None
    ):
        """
        Allows to configure spark's dynamic allocation

        :param min_executors:     Min. number of executors
        :param max_executors:     Max. number of executors
        :param initial_executors: Initial number of executors
        """
        self.spec.dynamic_allocation["enabled"] = True
        if min_executors:
            self.spec.dynamic_allocation["minExecutors"] = min_executors
        if max_executors:
            self.spec.dynamic_allocation["maxExecutors"] = max_executors
        if initial_executors:
            self.spec.dynamic_allocation["initialExecutors"] = initial_executors

    def disable_monitoring(self):
        self.spec.monitoring["enabled"] = False

    def _with_monitoring(self, enabled=True, exporter_jar=None):
        self.spec.monitoring["enabled"] = enabled
        if enabled:
            if exporter_jar:
                self.spec.monitoring["exporter_jar"] = exporter_jar

    def with_igz_spark(self, mount_v3io_to_executor=True):
        """
        Configures the pods (driver and executors) to have V3IO access (via file system and via Hadoop).

        :param mount_v3io_to_executor: When False, limits the file system mount to driver pod only. Default is True.
        """
        self._update_igz_jars(deps=self._get_igz_deps())
        self.apply(mount_v3io(name="v3io"))

        # if we only want to mount v3io on the driver, move v3io
        # mounts from common volume mounts to driver volume mounts
        if not mount_v3io_to_executor:
            v3io_mounts = []
            non_v3io_mounts = []
            for mount in self.spec.volume_mounts:
                if get_item_name(mount) == "v3io":
                    v3io_mounts.append(mount)
                else:
                    non_v3io_mounts.append(mount)
            self.spec.volume_mounts = non_v3io_mounts
            self.spec.driver_volume_mounts += v3io_mounts

        self.apply(
            mount_v3iod(
                namespace=config.namespace,
                v3io_config_configmap="spark-operator-v3io-config",
            )
        )
        if config.spark_history_server_path:
            self.spec.spark_conf["spark.eventLog.enabled"] = "true"
            self.spec.spark_conf["spark.eventLog.dir"] = (
                "file://" + config.spark_history_server_path
            )
        if "enabled" not in self.spec.monitoring or self.spec.monitoring["enabled"]:
            self._with_monitoring(
                exporter_jar="/spark/jars/jmx_prometheus_javaagent-0.16.1.jar",
            )

    def with_cores(
        self,
        executor_cores: typing.Optional[int] = None,
        driver_cores: typing.Optional[int] = None,
    ):
        """
        Allows to configure spark.executor.cores and spark.driver.cores parameters. The values must be integers
        greater than or equal to 1. If a parameter is not specified, it defaults to 1.

        Spark operator has multiple options to control the number of cores available to the executor and driver.
        The .coreLimit and .coreRequest parameters can be set for both executor and driver,
        but they only control the k8s properties of the pods created to run the driver/executor.
        Spark itself uses the spec.[executor|driver].cores parameter to set the parallelism of tasks and cores
        assigned to each task within the pod. This function sets the .cores parameters for the job executed.

        See https://github.com/kubeflow/spark-operator/issues/581 for a discussion about those
        parameters and their meaning in Spark operator.

        :param executor_cores: Number of cores to use for executor (spark.executor.cores)
        :param driver_cores:   Number of cores to use for driver (spark.driver.cores)
        """
        if executor_cores:
            if not isinstance(executor_cores, int) or executor_cores < 1:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"executor_cores must be an integer greater than or equal to 1. Got: {executor_cores}"
                )
            self.spec.executor_cores = executor_cores

        if driver_cores:
            if not isinstance(driver_cores, int) or driver_cores < 1:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"driver_cores must be an integer greater than or equal to 1. Got: {driver_cores}"
                )
            self.spec.driver_cores = driver_cores

    @classmethod
    def _get_default_deployed_mlrun_image_name(cls, with_gpu=False):
        suffix = cls.gpu_suffix if with_gpu else ""
        return cls.default_mlrun_image + suffix

    @classmethod
    def deploy_default_image(cls, with_gpu=False):
        sj = mlrun.new_function(kind=cls.kind, name="spark-default-image-deploy-temp")
        sj.spec.build.image = cls._get_default_deployed_mlrun_image_name(with_gpu)

        # setting required resources
        sj.with_executor_requests(cpu=1, mem="512m")
        sj.with_driver_requests(cpu=1, mem="512m")

        sj.deploy()
        mlrun.db.get_run_db().delete_function(name=sj.metadata.name)

    def _is_using_gpu(self):
        driver_limits = self.spec.driver_resources.get("limits")
        driver_gpu = None
        if driver_limits:
            _, driver_gpu = self._get_gpu_type_and_quantity(resources=driver_limits)

        executor_limits = self.spec.executor_resources.get("limits")
        executor_gpu = None
        if executor_limits:
            _, executor_gpu = self._get_gpu_type_and_quantity(resources=executor_limits)

        return bool(driver_gpu or executor_gpu)

    @property
    def _default_image(self):
        if config.spark_app_image_tag and config.spark_app_image:
            return (
                config.spark_app_image
                + (self.gpu_suffix if self._is_using_gpu() else "")
                + ":"
                + config.spark_app_image_tag
            )
        return None

    def deploy(
        self,
        watch=True,
        with_mlrun=True,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
        builder_env: typing.Optional[dict] = None,
        show_on_failure: bool = False,
        force_build: bool = False,
    ):
        """deploy function, build container with dependencies

        :param watch:                   wait for the deploy to complete (and print build logs)
        :param with_mlrun:              add the current mlrun package to the container build
        :param skip_deployed:           skip the build if we already have an image for the function
        :param is_kfp:                  deploy as part of a kfp pipeline
        :param mlrun_version_specifier: which mlrun package version to include (if not current)
        :param builder_env:             Kaniko builder pod env vars dict (for config/credentials)
                                        e.g. builder_env={"GIT_TOKEN": token}
        :param show_on_failure:         show logs only in case of build failure
        :param force_build:             set True for force building the image, even when no changes were made

        :return: True if the function is ready (deployed)
        """
        # connect will populate the config from the server config
        mlrun.db.get_run_db()
        if not self.spec.build.base_image:
            self.spec.build.base_image = self._default_image
        return super().deploy(
            watch=watch,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            is_kfp=is_kfp,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env=builder_env,
            show_on_failure=show_on_failure,
            force_build=force_build,
        )

    @staticmethod
    def _get_gpu_type_and_quantity(resources):
        gpu_type = [
            resource_type
            for resource_type in resources.keys()
            if resource_type not in ["cpu", "memory"]
        ]
        if len(gpu_type) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Sparkjob supports only a single gpu type"
            )
        gpu_quantity = resources[gpu_type[0]] if gpu_type else 0
        return gpu_type[0] if gpu_type else None, gpu_quantity

    def _pre_run(self, runobj: RunObject, execution: MLClientCtx):
        if self.spec.build.source and self.spec.build.load_source_on_run:
            raise mlrun.errors.MLRunPreconditionFailedError(
                "Sparkjob does not support loading source code on run, "
                "use func.with_source_archive(pull_at_runtime=False)"
            )

        super()._pre_run(runobj, execution)

    def _update_igz_jars(self, deps):
        if not self.spec.deps:
            self.spec.deps = {}
        if "jars" in deps:
            if "jars" not in self.spec.deps:
                self.spec.deps["jars"] = []
            self.spec.deps["jars"] += deps["jars"]
        if "files" in deps:
            if "files" not in self.spec.deps:
                self.spec.deps["files"] = []
            self.spec.deps["files"] += deps["files"]

    def with_limits(
        self,
        mem=None,
        cpu=None,
        gpus=None,
        gpu_type="nvidia.com/gpu",
        patch: bool = False,
    ):
        raise NotImplementedError(
            "In spark runtimes, use 'with_driver_limits' & 'with_executor_limits'"
        )

    def with_requests(self, mem=None, cpu=None, patch: bool = False):
        raise NotImplementedError(
            "In spark runtimes, use 'with_driver_requests' & 'with_executor_requests'"
        )

    def gpus(self, gpus, gpu_type="nvidia.com/gpu"):
        raise NotImplementedError(
            "In spark runtimes, use 'with_driver_limits' & 'with_executor_limits'"
        )

    def with_executor_requests(
        self,
        mem: typing.Optional[str] = None,
        cpu: typing.Optional[str] = None,
        patch: bool = False,
    ):
        """
        set executor pod required cpu/memory/gpu resources
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_requests("executor_resources", mem, cpu, patch=patch)

    def with_executor_limits(
        self,
        cpu: typing.Optional[str] = None,
        gpus: typing.Optional[int] = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set executor pod limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        # in spark operator there is only use of mem passed through requests,
        # limits is set to the same value so passing mem=None
        self.spec._verify_and_set_limits(
            "executor_resources", None, cpu, gpus, gpu_type, patch=patch
        )

    def with_driver_requests(
        self,
        mem: typing.Optional[str] = None,
        cpu: typing.Optional[str] = None,
        patch: bool = False,
    ):
        """
        set driver pod required cpu/memory/gpu resources
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec._verify_and_set_requests("driver_resources", mem, cpu, patch=patch)

    def with_driver_limits(
        self,
        cpu: typing.Optional[str] = None,
        gpus: typing.Optional[int] = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set driver pod cpu limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        # in spark operator there is only use of mem passed through requests,
        # limits is set to the same value so passing mem=None
        self.spec._verify_and_set_limits(
            "driver_resources", None, cpu, gpus, gpu_type, patch=patch
        )

    def with_restart_policy(
        self,
        restart_type="OnFailure",
        retries=0,
        retry_interval=10,
        submission_retries=3,
        submission_retry_interval=20,
    ):
        """set restart policy
        restart_type=OnFailure/Never/Always"""
        update_in(self.spec.restart_policy, "type", restart_type)
        update_in(self.spec.restart_policy, "retries", retries)
        update_in(self.spec.restart_policy, "retry_interval", retry_interval)
        update_in(self.spec.restart_policy, "submission_retries", submission_retries)
        update_in(
            self.spec.restart_policy,
            "submission_retry_interval",
            submission_retry_interval,
        )

    def with_source_archive(
        self, source, workdir=None, handler=None, pull_at_runtime=True, target_dir=None
    ):
        """load the code from git/tar/zip archive at runtime or build

        :param source:          valid path to git, zip, or tar file, e.g.
                                git://github.com/mlrun/something.git
                                http://some/url/file.zip
        :param handler:         default function handler
        :param workdir:         working dir relative to the archive root (e.g. './subdir') or absolute to the image root
        :param pull_at_runtime: not supported for spark runtime, must be False
        :param target_dir:      target dir on runtime pod for repo clone / archive extraction
        """
        if pull_at_runtime:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "pull_at_runtime is not supported for spark runtime, use pull_at_runtime=False"
            )

        super().with_source_archive(
            source, workdir, handler, pull_at_runtime, target_dir
        )

    def is_deployed(self):
        if (
            not self.spec.build.source
            and not self.spec.build.commands
            and not self.spec.build.extra
        ):
            return True
        return super().is_deployed()
