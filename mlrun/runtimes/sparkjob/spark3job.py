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

from ...utils import update_in, verify_and_update_in
from .abstract import AbstractSparkDefaults, AbstractSparkJobSpec, AbstractSparkRuntime


class Spark3Defaults(AbstractSparkDefaults):
    igz_deps = {
        "jars": [
            "local:///spark/v3io-libs/v3io-hcfs_2.12.jar",
            "local:///spark/v3io-libs/v3io-spark3-streaming_2.12.jar",
            "local:///spark/v3io-libs/v3io-spark3-object-dataframe_2.12.jar",
            "local:///igz/java/libs/scala-library-2.12.2.jar",
        ],
        "files": ["local:///igz/java/libs/v3io-pyspark.zip"],
    }
    spark_version = "3.1.2"


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
        use_default_image=False,
        priority_class_name=None,
        dynamic_allocation=None,
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


class Spark3Runtime(AbstractSparkRuntime):
    def _enrich_job(self, job):
        if self.spec.priority_class_name:
            verify_and_update_in(
                job,
                "spec.batchSchedulerOptions.priorityClassName",
                self.spec.priority_class_name,
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
        update_in(job, "spec.driver.serviceAccount", self._defaults.service_account)
        update_in(job, "spec.executor.serviceAccount", self._defaults.service_account)
        return

    @property
    def spec(self) -> Spark3JobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", Spark3JobSpec)

    @property
    def _defaults(self) -> Spark3Defaults:
        return Spark3Defaults
