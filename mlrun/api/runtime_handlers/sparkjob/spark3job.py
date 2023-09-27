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
import mlrun.utils.regex
from mlrun.api.runtime_handlers.sparkjob.abstract import AbstractSparkRuntimeHandler
from mlrun.utils import update_in, verify_and_update_in


class Spark3RuntimeHandler(AbstractSparkRuntimeHandler):
    @staticmethod
    def _enrich_job(
        runtime: mlrun.runtimes.sparkjob.spark3job.Spark3Runtime,
        job: dict,
    ):
        if runtime.spec.priority_class_name:
            verify_and_update_in(
                job,
                "spec.batchSchedulerOptions.priorityClassName",
                runtime.spec.priority_class_name,
                str,
            )

        verify_and_update_in(
            job,
            "spec.driver.cores",
            runtime.spec.driver_cores or 1,
            int,
        )
        # By default, we set this to 1 in the parent class. Here we override the value if requested.
        if runtime.spec.executor_cores:
            verify_and_update_in(
                job,
                "spec.executor.cores",
                runtime.spec.executor_cores,
                int,
            )

        if runtime.spec.dynamic_allocation:
            if "enabled" in runtime.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.enabled",
                    runtime.spec.dynamic_allocation["enabled"],
                )
            if "initialExecutors" in runtime.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.initialExecutors",
                    runtime.spec.dynamic_allocation["initialExecutors"],
                )
            if "minExecutors" in runtime.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.minExecutors",
                    runtime.spec.dynamic_allocation["minExecutors"],
                )
            if "maxExecutors" in runtime.spec.dynamic_allocation:
                update_in(
                    job,
                    "spec.dynamicAllocation.maxExecutors",
                    runtime.spec.dynamic_allocation["maxExecutors"],
                )
        update_in(job, "spec.driver.serviceAccount", "sparkapp")
        update_in(
            job,
            "spec.executor.serviceAccount",
            runtime.spec.service_account or "sparkapp",
        )
        if runtime.spec.driver_node_selector:
            update_in(
                job, "spec.driver.nodeSelector", runtime.spec.driver_node_selector
            )
        if runtime.spec.executor_node_selector:
            update_in(
                job, "spec.executor.nodeSelector", runtime.spec.executor_node_selector
            )
        if runtime.spec.driver_tolerations:
            update_in(job, "spec.driver.tolerations", runtime.spec.driver_tolerations)
        if runtime.spec.executor_tolerations:
            update_in(
                job, "spec.executor.tolerations", runtime.spec.executor_tolerations
            )

        if runtime.spec.driver_affinity:
            update_in(job, "spec.driver.affinity", runtime.spec.driver_affinity)
        if runtime.spec.executor_affinity:
            update_in(job, "spec.executor.affinity", runtime.spec.executor_affinity)

        if runtime.spec.monitoring:
            if (
                "enabled" in runtime.spec.monitoring
                and runtime.spec.monitoring["enabled"]
            ):
                update_in(job, "spec.monitoring.exposeDriverMetrics", True)
                update_in(job, "spec.monitoring.exposeExecutorMetrics", True)
                if "exporter_jar" in runtime.spec.monitoring:
                    update_in(
                        job,
                        "spec.monitoring.prometheus.jmxExporterJar",
                        runtime.spec.monitoring["exporter_jar"],
                    )

        if runtime.spec.driver_volume_mounts:
            update_in(
                job,
                "spec.driver.volumeMounts",
                runtime.spec.driver_volume_mounts,
                append=True,
            )
        if runtime.spec.executor_volume_mounts:
            update_in(
                job,
                "spec.executor.volumeMounts",
                runtime.spec.executor_volume_mounts,
                append=True,
            )
        if runtime.spec.driver_java_options:
            update_in(
                job,
                "spec.driver.javaOptions",
                runtime.spec.driver_java_options,
            )
        if runtime.spec.executor_java_options:
            update_in(
                job,
                "spec.executor.javaOptions",
                runtime.spec.executor_java_options,
            )
        return

    def _get_spark_version(self):
        return "3.1.2"
