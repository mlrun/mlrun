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
import abc
import os
import typing
from copy import deepcopy
from datetime import datetime
from typing import Optional

from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.utils.regex
import server.api.utils.singletons.k8s
from mlrun.common.runtimes.constants import RunStates, SparkApplicationStates
from mlrun.runtimes import RuntimeClassMode, Spark3Runtime
from mlrun.utils import (
    get_in,
    logger,
    update_in,
    verify_and_update_in,
    verify_field_regex,
    verify_list_and_update_in,
)
from server.api.db.base import DBInterface
from server.api.runtime_handlers.kubejob import KubeRuntimeHandler

_sparkjob_template = {
    "apiVersion": "sparkoperator.k8s.io/v1beta2",
    "kind": "SparkApplication",
    "metadata": {"name": "", "namespace": "default-tenant"},
    "spec": {
        "mode": "cluster",
        "image": "",
        "mainApplicationFile": "",
        "sparkVersion": "3.1.2",
        "restartPolicy": {
            "type": "OnFailure",
            "onFailureRetries": 0,
            "onFailureRetryInterval": 10,
            "onSubmissionFailureRetries": 3,
            "onSubmissionFailureRetryInterval": 20,
        },
        "deps": {},
        "volumes": [],
        "driver": {
            "cores": 1,
            "coreLimit": "1200m",
            "memory": "512m",
            "labels": {},
            "volumeMounts": [],
            "env": [],
        },
        "executor": {
            "cores": 0,
            "instances": 0,
            "memory": "",
            "labels": {},
            "volumeMounts": [],
            "env": [],
        },
    },
}


class Spark3RuntimeHandler(KubeRuntimeHandler, abc.ABC):
    kind = "spark"
    class_modes = {
        RuntimeClassMode.run: "spark",
    }

    def run(
        self,
        runtime: mlrun.runtimes.sparkjob.Spark3Runtime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
    ):
        self._validate_sparkjob(runtime, run)

        if run.metadata.iteration:
            runtime.store_run(run)
        job = deepcopy(_sparkjob_template)
        meta = self._get_meta(runtime, run, True)
        pod_labels = deepcopy(meta.labels)
        pod_labels[mlrun_constants.MLRunInternalLabels.job] = meta.name
        job_type = runtime.spec.job_type or "Python"
        update_in(job, "spec.type", job_type)
        if runtime.spec.job_type == "Python":
            update_in(job, "spec.pythonVersion", runtime.spec.python_version or "3")
        if runtime.spec.main_class:
            update_in(job, "spec.mainClass", runtime.spec.main_class)
        update_in(
            job,
            "spec.sparkVersion",
            runtime.spec.spark_version or self._get_spark_version(),
        )

        if runtime.spec.image_pull_policy:
            verify_and_update_in(
                job, "spec.imagePullPolicy", runtime.spec.image_pull_policy, str
            )

        if runtime.spec.restart_policy:
            verify_and_update_in(
                job, "spec.restartPolicy.type", runtime.spec.restart_policy["type"], str
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onFailureRetries",
                runtime.spec.restart_policy["retries"],
                int,
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onFailureRetryInterval",
                runtime.spec.restart_policy["retry_interval"],
                int,
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onSubmissionFailureRetries",
                runtime.spec.restart_policy["submission_retries"],
                int,
            )
            verify_and_update_in(
                job,
                "spec.restartPolicy.onSubmissionFailureRetryInterval",
                runtime.spec.restart_policy["submission_retry_interval"],
                int,
            )

        update_in(job, "metadata", meta.to_dict())
        update_in(job, "spec.driver.labels", pod_labels)
        update_in(job, "spec.executor.labels", pod_labels)
        verify_and_update_in(
            job,
            "spec.executor.instances",
            runtime.spec.replicas or 1,
            int,
        )
        if runtime.spec.image_pull_secret:
            update_in(job, "spec.imagePullSecrets", [runtime.spec.image_pull_secret])

        if run.spec.node_selector:
            update_in(
                job,
                "spec.nodeSelector",
                None,
            )

        if not runtime.spec.image:
            if runtime.spec.use_default_image:
                runtime.spec.image = runtime._get_default_deployed_mlrun_image_name(
                    runtime._is_using_gpu()
                )
            elif runtime._default_image:
                runtime.spec.image = runtime._default_image

        update_in(
            job,
            "spec.image",
            runtime.full_image_path(
                client_version=run.metadata.labels.get(
                    mlrun_constants.MLRunInternalLabels.client_version
                ),
                client_python_version=run.metadata.labels.get(
                    mlrun_constants.MLRunInternalLabels.client_python_version
                ),
            ),
        )

        update_in(job, "spec.volumes", runtime.spec.volumes)

        self.add_secrets_to_spec_before_running(
            runtime, project_name=run.metadata.project
        )

        command, args, extra_env = self._get_cmd_args(runtime, run)
        code = None
        if "MLRUN_EXEC_CODE" in [e.get("name") for e in extra_env]:
            code = f"""
import mlrun.__main__ as ml
ctx = ml.main.make_context('main', {args})
with ctx:
    result = ml.main.invoke(ctx)
"""

        update_in(job, "spec.driver.env", extra_env + runtime.spec.env)
        update_in(job, "spec.executor.env", extra_env + runtime.spec.env)
        update_in(job, "spec.driver.volumeMounts", runtime.spec.volume_mounts)
        update_in(job, "spec.executor.volumeMounts", runtime.spec.volume_mounts)
        update_in(job, "spec.deps", runtime.spec.deps)

        spark_conf = runtime.spec.spark_conf
        if spark_conf:
            if (
                spark_conf.get("spark.eventLog.enabled")
                and "spark.eventLog.dir" not in spark_conf
            ):
                spark_conf["spark.eventLog.dir"] = "/tmp"

            job["spec"]["sparkConf"] = {}
            for k, v in spark_conf.items():
                job["spec"]["sparkConf"][f"{k}"] = f"{v}"

        if runtime.spec.hadoop_conf:
            job["spec"]["hadoopConf"] = {}
            for k, v in runtime.spec.hadoop_conf.items():
                job["spec"]["hadoopConf"][f"{k}"] = f"{v}"

        executor_cpu_limit = None
        if "limits" in runtime.spec.executor_resources:
            if "cpu" in runtime.spec.executor_resources["limits"]:
                executor_cpu_limit = runtime.spec.executor_resources["limits"]["cpu"]
                verify_and_update_in(
                    job,
                    "spec.executor.coreLimit",
                    executor_cpu_limit,
                    str,
                )
        if "requests" in runtime.spec.executor_resources:
            verify_and_update_in(
                job,
                "spec.executor.cores",
                1,  # Must be set due to CRD validations. Will be overridden by coreRequest
                int,
            )
            if "cpu" in runtime.spec.executor_resources["requests"]:
                if executor_cpu_limit is not None:
                    executor_cpu_request = runtime.spec.executor_resources["requests"][
                        "cpu"
                    ]
                    if self._parse_cpu_resource_string(
                        executor_cpu_request
                    ) > self._parse_cpu_resource_string(executor_cpu_limit):
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            f"Executor CPU request ({executor_cpu_request}) is higher than limit "
                            f"({executor_cpu_limit})"
                        )
                verify_and_update_in(
                    job,
                    "spec.executor.coreRequest",
                    str(
                        runtime.spec.executor_resources["requests"]["cpu"]
                    ),  # Backwards compatibility
                    str,
                )
            if "memory" in runtime.spec.executor_resources["requests"]:
                verify_and_update_in(
                    job,
                    "spec.executor.memory",
                    runtime.spec.executor_resources["requests"]["memory"],
                    str,
                )
            gpu_type, gpu_quantity = runtime._get_gpu_type_and_quantity(
                resources=runtime.spec.executor_resources["limits"]
            )
            if gpu_type:
                update_in(job, "spec.executor.gpu.name", gpu_type)
                if gpu_quantity:
                    verify_and_update_in(
                        job,
                        "spec.executor.gpu.quantity",
                        gpu_quantity,
                        int,
                    )
        driver_cpu_limit = None
        if "limits" in runtime.spec.driver_resources:
            if "cpu" in runtime.spec.driver_resources["limits"]:
                driver_cpu_limit = runtime.spec.driver_resources["limits"]["cpu"]
                verify_and_update_in(
                    job,
                    "spec.driver.coreLimit",
                    runtime.spec.driver_resources["limits"]["cpu"],
                    str,
                )
        if "requests" in runtime.spec.driver_resources:
            if "cpu" in runtime.spec.driver_resources["requests"]:
                if driver_cpu_limit is not None:
                    driver_cpu_request = runtime.spec.driver_resources["requests"][
                        "cpu"
                    ]
                    if self._parse_cpu_resource_string(
                        driver_cpu_request
                    ) > self._parse_cpu_resource_string(driver_cpu_limit):
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            f"Driver CPU request ({driver_cpu_request}) is higher than limit "
                            f"({driver_cpu_limit})"
                        )
                verify_and_update_in(
                    job,
                    "spec.driver.coreRequest",
                    str(runtime.spec.driver_resources["requests"]["cpu"]),
                    str,
                )
            if "memory" in runtime.spec.driver_resources["requests"]:
                verify_and_update_in(
                    job,
                    "spec.driver.memory",
                    runtime.spec.driver_resources["requests"]["memory"],
                    str,
                )
            gpu_type, gpu_quantity = runtime._get_gpu_type_and_quantity(
                resources=runtime.spec.driver_resources["limits"]
            )
            if gpu_type:
                update_in(job, "spec.driver.gpu.name", gpu_type)
                if gpu_quantity:
                    verify_and_update_in(
                        job,
                        "spec.driver.gpu.quantity",
                        gpu_quantity,
                        int,
                    )

        self._enrich_job(runtime, job)

        self._enrich_node_selectors(run, runtime, job, run.metadata.project)

        if runtime.spec.command:
            if "://" not in runtime.spec.command:
                workdir = self._resolve_workdir(runtime)
                runtime.spec.command = "local://" + os.path.join(
                    workdir or "",
                    runtime.spec.command,
                )
            update_in(job, "spec.mainApplicationFile", runtime.spec.command)

        verify_list_and_update_in(job, "spec.arguments", runtime.spec.args or [], str)
        self._submit_spark_job(runtime, job, meta, code)

    @staticmethod
    def _submit_spark_job(
        runtime: mlrun.runtimes.sparkjob.Spark3Runtime,
        job: dict,
        meta: k8s_client.V1ObjectMeta,
        code: Optional[str] = None,
    ):
        namespace = meta.namespace
        k8s = server.api.utils.singletons.k8s.get_k8s_helper()
        namespace = k8s.resolve_namespace(namespace)
        if code:
            k8s_config_map = k8s_client.V1ConfigMap()
            k8s_config_map.metadata = meta
            k8s_config_map.metadata.name += "-script"
            k8s_config_map.data = {runtime.code_script: code}
            config_map = k8s.v1api.create_namespaced_config_map(
                namespace, k8s_config_map
            )
            config_map_name = config_map.metadata.name

            vol_src = k8s_client.V1ConfigMapVolumeSource(name=config_map_name)
            volume_name = "script"
            vol = k8s_client.V1Volume(name=volume_name, config_map=vol_src)
            vol_mount = k8s_client.V1VolumeMount(
                mount_path=runtime.code_path, name=volume_name
            )
            update_in(job, "spec.volumes", [vol], append=True)
            update_in(job, "spec.driver.volumeMounts", [vol_mount], append=True)
            update_in(job, "spec.executor.volumeMounts", [vol_mount], append=True)
            update_in(
                job,
                "spec.mainApplicationFile",
                f"local://{runtime.code_path}/{runtime.code_script}",
            )

        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                Spark3Runtime.group,
                Spark3Runtime.version,
                namespace=namespace,
                plural=Spark3Runtime.plural,
                body=job,
            )
            name = get_in(resp, "metadata.name", "unknown")
            logger.info(f"SparkJob {name} created")
            return resp
        except ApiException as exc:
            crd = (
                f"{Spark3Runtime.group}/{Spark3Runtime.version}/{Spark3Runtime.plural}"
            )
            logger.error(
                f"Exception when creating SparkJob ({crd}): {mlrun.errors.err_to_str(exc)}"
            )
            raise mlrun.runtimes.RunError("Exception when creating SparkJob") from exc

    @staticmethod
    def _validate_sparkjob(
        runtime: mlrun.runtimes.sparkjob.Spark3Runtime,
        run: mlrun.run.RunObject,
    ):
        # validating correctness of sparkjob's function name
        try:
            verify_field_regex(
                "run.metadata.name",
                run.metadata.name,
                mlrun.utils.regex.sparkjob_name,
            )

        except mlrun.errors.MLRunInvalidArgumentError as err:
            pattern_error = str(err).split(" ")[-1]
            if pattern_error == mlrun.utils.regex.sprakjob_length:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Job name '{run.metadata.name}' is not valid."
                    f" The job name must be not longer than 29 characters"
                )
            elif pattern_error in mlrun.utils.regex.label_value:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "a valid label must be an empty string or consist of alphanumeric characters,"
                    " '-', '_' or '.', and must start and end with an alphanumeric character"
                )
            elif pattern_error in mlrun.utils.regex.sparkjob_service_name:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "a valid label must consist of lower case alphanumeric characters or '-', start with "
                    "an alphabetic character, and end with an alphanumeric character"
                )
            else:
                raise err

        # validating existence of required fields
        if "requests" not in runtime.spec.executor_resources:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Sparkjob must contain executor requests"
            )
        if "requests" not in runtime.spec.driver_resources:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Sparkjob must contain driver requests"
            )

    @staticmethod
    def _parse_cpu_resource_string(cpu):
        if isinstance(cpu, str) and cpu.endswith("m"):
            return float(cpu[:-1]) / 1000
        else:
            return float(cpu)

    def _resolve_crd_object_status_info(
        self, crd_object: dict
    ) -> tuple[bool, Optional[datetime], Optional[str]]:
        state = crd_object.get("status", {}).get("applicationState", {}).get("state")
        if not state:
            return False, None, None
        in_terminal_state = state in SparkApplicationStates.terminal_states()
        desired_run_state = SparkApplicationStates.spark_application_state_to_run_state(
            state
        )
        completion_time = None
        if in_terminal_state:
            if crd_object.get("status", {}).get("terminationTime"):
                completion_time = datetime.fromisoformat(
                    crd_object.get("status", {})
                    .get("terminationTime")
                    .replace("Z", "+00:00")
                )
            else:
                last_submission_attempt_time = crd_object.get("status", {}).get(
                    "lastSubmissionAttemptTime"
                )
                if last_submission_attempt_time:
                    last_submission_attempt_time = last_submission_attempt_time.replace(
                        "Z", "+00:00"
                    )
                    completion_time = datetime.fromisoformat(
                        last_submission_attempt_time
                    )
        return in_terminal_state, completion_time, desired_run_state

    def _resolve_container_error_status(self, crd_object: dict) -> tuple[str, str]:
        error_message = (
            crd_object.get("status", {})
            .get("applicationState", {})
            .get("errorMessage", "")
        )
        return "", error_message

    def _is_terminal_state(self, runtime_resource: dict) -> bool:
        state = (
            runtime_resource.get("status", {}).get("applicationState", {}).get("state")
        )
        return state in SparkApplicationStates.terminal_states()

    def _update_ui_url(
        self,
        db: DBInterface,
        db_session: Session,
        project: str,
        uid: str,
        crd_object,
        run: dict,
    ):
        if not run:
            logger.warning(
                "Run object was not provided, cannot update the UI URL",
                project=project,
                uid=uid,
                run=run,
            )
            return

        app_state = (
            crd_object.get("status", {}).get("applicationState", {}).get("state")
        )
        state = SparkApplicationStates.spark_application_state_to_run_state(app_state)
        ui_url = None
        if state == RunStates.running:
            ui_url = (
                crd_object.get("status", {})
                .get("driverInfo", {})
                .get("webUIIngressAddress")
            )

        db_ui_url = run.get("status", {}).get("ui_url")
        if db_ui_url == ui_url:
            return

        run.setdefault("status", {})["ui_url"] = ui_url
        db.store_run(db_session, run, uid, project)

    @staticmethod
    def are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"{mlrun_constants.MLRunInternalLabels.uid}={object_id}"

    @staticmethod
    def _get_main_runtime_resource_label_selector() -> str:
        """
        There are some runtimes which might have multiple k8s resources attached to a one runtime, in this case
        we don't want to pull logs from all but rather only for the "driver"/"launcher" etc
        :return: the label selector
        """
        return "spark-role=driver"

    @staticmethod
    def _get_crd_info() -> tuple[str, str, str]:
        return (
            Spark3Runtime.group,
            Spark3Runtime.version,
            Spark3Runtime.plural,
        )

    def _delete_extra_resources(
        self,
        db: DBInterface,
        db_session: Session,
        namespace: str,
        deleted_resources: list[dict],
        label_selector: str = None,
        force: bool = False,
        grace_period: int = None,
        resource_deletion_grace_period: typing.Optional[int] = None,
    ):
        """
        Handling config maps deletion
        """
        uids = []
        for crd_dict in deleted_resources:
            uid = (
                crd_dict["metadata"]
                .get("labels", {})
                .get(mlrun_constants.MLRunInternalLabels.uid, None)
            )
            uids.append(uid)

        config_maps = server.api.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_config_map(
            namespace, label_selector=label_selector
        )
        for config_map in config_maps.items:
            try:
                uid = config_map.metadata.labels.get(
                    mlrun_constants.MLRunInternalLabels.uid, None
                )
                if force or uid in uids:
                    server.api.utils.singletons.k8s.get_k8s_helper().v1api.delete_namespaced_config_map(
                        config_map.metadata.name,
                        namespace,
                        grace_period_seconds=resource_deletion_grace_period,
                    )
                    logger.info(f"Deleted config map: {config_map.metadata.name}")
            except ApiException as exc:
                # ignore error if config map is already removed
                if exc.status != 404:
                    raise

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

    def _enrich_node_selectors(
        self,
        run: mlrun.run.RunObject,
        runtime: mlrun.runtimes.sparkjob.spark3job.Spark3Runtime,
        job: dict,
        project_name: str,
    ):
        # If `driver_node_selector` is defined, it takes precedence and will be used as the driver node selector.
        # If not provided or empty, the `node_selector` from the run object, which is enriched from the runtime object
        # when the run is launched, will be used instead.
        # The same logic applies to `executor_node_selector`.
        driver_node_selector = (
            runtime.spec.driver_node_selector or run.spec.node_selector
        )

        executor_node_selector = (
            runtime.spec.executor_node_selector or run.spec.node_selector
        )

        if driver_node_selector:
            update_in(job, "spec.driver.nodeSelector", driver_node_selector)
        if executor_node_selector:
            update_in(job, "spec.executor.nodeSelector", executor_node_selector)

        # we enrich the node selectors with any additional selectors,
        # from the project-level configuration and global config, if exists.
        project_node_selector = {}
        config_node_selector = mlrun.mlconf.get_default_function_node_selector()
        if project_name:
            if project := runtime._get_db().get_project(project_name):
                project_node_selector = project.spec.default_function_node_selector

        if project_node_selector or config_node_selector:
            logger.info(
                "Enriching driver and executor node selector from mlrun project and config",
                config_node_selector=config_node_selector,
                project_node_selector=project_node_selector,
            )
            update_in(
                job,
                "spec.driver.nodeSelector",
                mlrun.utils.helpers.to_non_empty_values_dict(
                    mlrun.utils.helpers.merge_dicts_with_precedence(
                        config_node_selector,
                        project_node_selector,
                        driver_node_selector,
                    )
                ),
            )
            update_in(
                job,
                "spec.executor.nodeSelector",
                mlrun.utils.helpers.to_non_empty_values_dict(
                    mlrun.utils.helpers.merge_dicts_with_precedence(
                        config_node_selector,
                        project_node_selector,
                        executor_node_selector,
                    )
                ),
            )

    def _get_spark_version(self):
        return "3.2.3"
