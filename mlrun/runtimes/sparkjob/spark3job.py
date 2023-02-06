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

import typing

import kubernetes.client

import mlrun.api.schemas.function
import mlrun.errors
import mlrun.runtimes.pod

from ...utils import update_in, verify_and_update_in
from .abstract import AbstractSparkJobSpec, AbstractSparkRuntime


class Spark3JobSpec(AbstractSparkJobSpec):
    # https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/55732a6a392cbe1d6546c7ec6823193ab055d2fa/pkg/apis/sparkoperator.k8s.io/v1beta2/types.go#L181
    _dict_fields = AbstractSparkJobSpec._dict_fields + [
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

    def to_dict(self, fields=None, exclude=None):
        exclude = exclude or []
        _exclude = [
            "affinity",
            "tolerations",
            "security_context",
            "executor_affinity",
            "executor_tolerations",
            "driver_affinity",
            "driver_tolerations",
        ]
        struct = super().to_dict(fields, exclude=list(set(exclude + _exclude)))
        api = kubernetes.client.ApiClient()
        for field in _exclude:
            if field not in exclude:
                struct[field] = api.sanitize_for_serialization(getattr(self, field))
        return struct

    @property
    def executor_tolerations(self) -> typing.List[kubernetes.client.V1Toleration]:
        return self._executor_tolerations

    @executor_tolerations.setter
    def executor_tolerations(self, executor_tolerations):
        self._executor_tolerations = (
            mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                "executor_tolerations", executor_tolerations
            )
        )

    @property
    def driver_tolerations(self) -> typing.List[kubernetes.client.V1Toleration]:
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


class Spark3Runtime(AbstractSparkRuntime):
    def _enrich_job(self, job):
        if self.spec.priority_class_name:
            verify_and_update_in(
                job,
                "spec.batchSchedulerOptions.priorityClassName",
                self.spec.priority_class_name,
                str,
            )

        verify_and_update_in(
            job,
            "spec.driver.cores",
            self.spec.driver_cores or 1,
            int,
        )
        # By default we set this to 1 in the parent class. Here we override the value if requested.
        if self.spec.executor_cores:
            verify_and_update_in(
                job,
                "spec.executor.cores",
                self.spec.executor_cores,
                int,
            )

        if "requests" in self.spec.driver_resources:
            if "cpu" in self.spec.driver_resources["requests"]:
                verify_and_update_in(
                    job,
                    "spec.driver.coreRequest",
                    str(self.spec.driver_resources["requests"]["cpu"]),
                    str,
                )

        if self.spec.dynamic_allocation:
            if "enabled" in self.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.enabled",
                    self.spec.dynamic_allocation["enabled"],
                )
            if "initialExecutors" in self.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.initialExecutors",
                    self.spec.dynamic_allocation["initialExecutors"],
                )
            if "minExecutors" in self.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.minExecutors",
                    self.spec.dynamic_allocation["minExecutors"],
                )
            if "maxExecutors" in self.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.maxExecutors",
                    self.spec.dynamic_allocation["maxExecutors"],
                )
        update_in(job, "spec.driver.serviceAccount", "sparkapp")
        update_in(
            job, "spec.executor.serviceAccount", self.spec.service_account or "sparkapp"
        )
        if self.spec.driver_node_selector:
            update_in(job, "spec.driver.nodeSelector", self.spec.driver_node_selector)
        if self.spec.executor_node_selector:
            update_in(
                job, "spec.executor.nodeSelector", self.spec.executor_node_selector
            )
        if self.spec.driver_tolerations:
            update_in(job, "spec.driver.tolerations", self.spec.driver_tolerations)
        if self.spec.executor_tolerations:
            update_in(job, "spec.executor.tolerations", self.spec.executor_tolerations)

        if self.spec.driver_affinity:
            update_in(job, "spec.driver.affinity", self.spec.driver_affinity)
        if self.spec.executor_affinity:
            update_in(job, "spec.executor.affinity", self.spec.executor_affinity)

        if self.spec.monitoring:
            if "enabled" in self.spec.monitoring and self.spec.monitoring["enabled"]:
                update_in(job, "spec.monitoring.exposeDriverMetrics", True)
                update_in(job, "spec.monitoring.exposeExecutorMetrics", True)
                if "exporter_jar" in self.spec.monitoring:
                    update_in(
                        job,
                        "spec.monitoring.prometheus.jmxExporterJar",
                        self.spec.monitoring["exporter_jar"],
                    )

        if self.spec.driver_volume_mounts:
            update_in(
                job,
                "spec.driver.volumeMounts",
                self.spec.driver_volume_mounts,
                append=True,
            )
        if self.spec.executor_volume_mounts:
            update_in(
                job,
                "spec.executor.volumeMounts",
                self.spec.executor_volume_mounts,
                append=True,
            )
        if self.spec.driver_java_options:
            update_in(
                job,
                "spec.driver.javaOptions",
                self.spec.driver_java_options,
            )
        if self.spec.executor_java_options:
            update_in(
                job,
                "spec.executor.javaOptions",
                self.spec.executor_java_options,
            )
        return

    def _get_spark_version(self):
        return "3.1.2"

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

    @property
    def spec(self) -> Spark3JobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", Spark3JobSpec)

    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[kubernetes.client.V1Affinity] = None,
        tolerations: typing.Optional[
            typing.List[kubernetes.client.V1Toleration]
        ] = None,
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
                "Tolerations can be set in spark runtime but not in with_node_selection"
                "Instead, use with_driver_node_selection and with_executor_node_selection to set tolerations"
            )
        super().with_node_selection(node_name, node_selector, affinity, tolerations)

    def with_driver_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[kubernetes.client.V1Affinity] = None,
        tolerations: typing.Optional[
            typing.List[kubernetes.client.V1Toleration]
        ] = None,
    ):
        """
        Enables to control on which k8s node the spark executor will run

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels will be eligible to be picked
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
        if affinity:
            self.spec.driver_affinity = affinity
        if node_selector:
            self.spec.driver_node_selector = node_selector
        if tolerations:
            self.spec.driver_tolerations = tolerations

    def with_executor_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[kubernetes.client.V1Affinity] = None,
        tolerations: typing.Optional[
            typing.List[kubernetes.client.V1Toleration]
        ] = None,
    ):
        """
        Enables to control on which k8s node the spark executor will run

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels will be eligible to be picked
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
        if affinity:
            self.spec.executor_affinity = affinity
        if node_selector:
            self.spec.executor_node_selector = node_selector
        if tolerations:
            self.spec.executor_tolerations = tolerations

    def with_preemption_mode(
        self, mode: typing.Union[mlrun.api.schemas.function.PreemptionModes, str]
    ):
        """
        Use with_driver_preemption_mode / with_executor_preemption_mode to setup preemption_mode for spark operator
        """
        raise mlrun.errors.MLRunInvalidArgumentTypeError(
            "with_preemption_mode is not supported use with_driver_preemption_mode / with_executor_preemption_mode"
            " to set preemption mode for spark operator"
        )

    def with_driver_preemption_mode(
        self, mode: typing.Union[mlrun.api.schemas.function.PreemptionModes, str]
    ):
        """
        Preemption mode controls whether the spark driver can be scheduled on preemptible nodes.
        Tolerations, node selector, and affinity are populated on preemptible nodes corresponding to the function spec.

        The supported modes are:

        * **allow** - The function can be scheduled on preemptible nodes
        * **constrain** - The function can only run on preemptible nodes
        * **prevent** - The function cannot be scheduled on preemptible nodes
        * **none** - No preemptible configuration will be applied on the function

        The default preemption mode is configurable in mlrun.mlconf.function_defaults.preemption_mode,
        by default it's set to **prevent**

        :param mode: allow | constrain | prevent | none defined in :py:class:`~mlrun.api.schemas.PreemptionModes`
        """
        preemption_mode = mlrun.api.schemas.function.PreemptionModes(mode)
        self.spec.driver_preemption_mode = preemption_mode.value

    def with_executor_preemption_mode(
        self, mode: typing.Union[mlrun.api.schemas.function.PreemptionModes, str]
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

        :param mode: allow | constrain | prevent | none defined in :py:class:`~mlrun.api.schemas.PreemptionModes`
        """
        preemption_mode = mlrun.api.schemas.function.PreemptionModes(mode)
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
        Add an host path volume and mount it to the driver pod
        More info: https://kubernetes.io/docs/concepts/storage/volumes#hostpath

        :param host_path:   Path of the directory on the host. If the path is a symlink, it will follow the link to the
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

        :param host_path:   Path of the directory on the host. If the path is a symlink, it will follow the link to the
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
        super().with_igz_spark(mount_v3io_to_executor)
        if "enabled" not in self.spec.monitoring or self.spec.monitoring["enabled"]:
            self._with_monitoring(
                exporter_jar="/spark/jars/jmx_prometheus_javaagent-0.16.1.jar",
            )

    def with_cores(self, executor_cores: int = None, driver_cores: int = None):
        """
        Allows to configure spark.executor.cores and spark.driver.cores parameters. The values must be integers
        greater than or equal to 1. If a parameter is not specified, it defaults to 1.

        Spark operator has multiple options to control the number of cores available to the executor and driver.
        The .coreLimit and .coreRequest parameters can be set for both executor and driver,
        but they only controls the k8s properties of the pods created to run driver/executor.
        Spark itself uses the spec.[executor|driver].cores parameter to set the parallelism of tasks and cores
        assigned to each task within the pod. This function sets the .cores parameters for the job executed.

        See https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/issues/581 for a discussion about those
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
