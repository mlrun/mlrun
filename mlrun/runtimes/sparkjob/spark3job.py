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

from kubernetes import client

from ...utils import update_in, verify_and_update_in
from .abstract import AbstractSparkJobSpec, AbstractSparkRuntime


class Spark3JobSpec(AbstractSparkJobSpec):
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
            1,  # Must be set due to CRD validations. Will be overridden by coreRequest
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
        update_in(job, "spec.executor.serviceAccount", "sparkapp")
        if self.spec.driver_node_selector:
            update_in(job, "spec.driver.nodeSelector", self.spec.driver_node_selector)
        if self.spec.executor_node_selector:
            update_in(
                job, "spec.executor.nodeSelector", self.spec.executor_node_selector
            )
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

    def with_driver_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
    ):
        """
        Enables to control on which k8s node the spark executor will run

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels will be eligible to be picked
        :param affinity:        Expands the types of constraints you can express - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
                                for details
        """
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        if affinity:
            raise NotImplementedError(
                "Setting affinity is not supported for spark runtime"
            )

        if node_selector:
            self.spec.driver_node_selector = node_selector

    def with_executor_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
    ):
        """
        Enables to control on which k8s node the spark executor will run

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels will be eligible to be picked
        :param affinity:        Expands the types of constraints you can express - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
                                for details
        """
        if node_name:
            raise NotImplementedError(
                "Setting node name is not supported for spark runtime"
            )
        if affinity:
            raise NotImplementedError(
                "Setting affinity is not supported for spark runtime"
            )
        if node_selector:
            self.spec.executor_node_selector = node_selector

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

    def with_igz_spark(self):
        super().with_igz_spark()
        if "enabled" not in self.spec.monitoring or self.spec.monitoring["enabled"]:
            self._with_monitoring(
                exporter_jar="/spark/jars/jmx_prometheus_javaagent-0.16.1.jar",
            )
