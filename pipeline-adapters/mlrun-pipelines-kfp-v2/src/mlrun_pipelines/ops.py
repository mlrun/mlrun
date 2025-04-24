# Copyright 2024 Iguazio
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

import os
from typing import Optional

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
import mlrun.utils.helpers
import mlrun_pipelines.common.ops
from mlrun.config import config
from mlrun.utils import get_in, logger
from mlrun_pipelines.common.constants import PipelineRunType
from mlrun_pipelines.common.helpers import (
    FUNCTION_ANNOTATION,
    PROJECT_ANNOTATION,
    RUN_ANNOTATION,
)
from mlrun_pipelines.imports import dsl
from mlrun_pipelines.imports import kubernetes as kfp_k8s


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.workflow_manifest()
    if not workflow:
        return None, project, None

    templates = {}
    for name, template in workflow.get_executors():
        project = project or get_in(
            template, ["metadata", "annotations", PROJECT_ANNOTATION], ""
        )
        templates[name] = {
            "run_type": get_in(
                template, ["metadata", "annotations", RUN_ANNOTATION], ""
            ),
            "function": get_in(
                template, ["metadata", "annotations", FUNCTION_ANNOTATION], ""
            ),
        }

    dag = {}
    nodes = []
    if run["run_details"]:
        nodes = run["run_details"].get("task_details", [])
    for node in nodes:
        name = (
            node["display_name"]
            if not node["child_tasks"]
            else node["child_tasks"][0]["pod_name"]
        )
        if not name:
            continue
        record = {
            "phase": node["state"],
            "started_at": node["create_time"],
            "finished_at": node["end_time"],
            "id": node["task_id"],
            "parent": node.get("parent_task_id", ""),
            "name": node["display_name"],
            "type": "DAG" if node["child_tasks"] else "Pod",
            "children": [c["pod_name"] for c in node["child_tasks"] or []],
        }

        if name in templates:
            record["function"] = templates[name].get("function")
            record["run_type"] = templates[name].get("run_type")
        dag[name] = record

    return dag, project, run["state"]


def add_default_function_resources(
    task,
    function: dsl.PipelineTask,
) -> dsl.PipelineTask:
    __set_task_requests = {
        "cpu": task.set_cpu_request,
        "memory": task.set_memory_request,
    }
    __set_task_limits = {
        "cpu": task.set_cpu_limit,
        "memory": task.set_memory_limit,
    }

    default_resources = config.get_default_function_pod_resources()
    for resource_name, resource_value in default_resources["requests"].items():
        if resource_value:
            __set_task_requests[resource_name](resource_value)

    for resource_name, resource_value in default_resources["limits"].items():
        if resource_value:
            __set_task_limits[resource_name](resource_value)

    mlrun_pipelines.common.ops._enrich_gpu_limits(function=function, task=task)
    return task


def add_function_node_selection_attributes(
    function, task: dsl.PipelineTask
) -> dsl.PipelineTask:
    if not mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind):
        enriched_node_selector = mlrun_pipelines.common.ops._enrich_node_selector(
            function
        )
        if enriched_node_selector:
            for k, v in enriched_node_selector.items():
                task = kfp_k8s.add_node_selector(task, k, v)

    if getattr(function.spec, "tolerations"):
        if hasattr(kfp_k8s, "add_toleration"):
            for t in function.spec.tolerations:
                task = kfp_k8s.add_toleration(
                    task,
                    t.key,
                    t.operator,
                    t.value,
                    t.effect,
                    t.toleration_seconds,
                )
        else:
            # TODO: remove this warning as soon as KFP SDK >=2.7.0 is available for MLRun SDK
            logger.warning(
                "Support for Pod tolerations is not yet available on the KFP 2 engine",
                project=function.metadata.project,
                function_name=function.metadata.name,
            )

    # TODO: remove this warning as soon as KFP SDK provides support for affinity management
    if getattr(function.spec, "affinity"):
        logger.warning(
            "Support for Pod affinity is not yet available on the KFP 2 engine",
            project=function.metadata.project,
            function_name=function.metadata.name,
        )

    return task


def add_annotations(
    task: dsl.PipelineTask,
    kind: str,
    function,
    func_url: Optional[str] = None,
    project: Optional[str] = None,
):
    # TODO: remove this warning as soon as KFP SDK >=2.7.0 is available for MLRun SDK
    if not hasattr(kfp_k8s, "add_pod_annotation"):
        logger.warning(
            "Support for Pod annotations is not yet available on the KFP 2 engine",
            project=project,
            function_name=function.metadata.name,
        )
        return

    if func_url and func_url.startswith("db://"):
        func_url = func_url[len("db://") :]
    kfp_k8s.add_pod_annotation(task, RUN_ANNOTATION, kind)
    kfp_k8s.add_pod_annotation(
        task, PROJECT_ANNOTATION, project or function.metadata.project
    )
    kfp_k8s.add_pod_annotation(task, FUNCTION_ANNOTATION, func_url or function.uri)
    return task


def add_labels(task, function, scrape_metrics=False):
    # TODO: remove this warning as soon as KFP SDK >=2.7.0 is available for MLRun SDK
    if not hasattr(kfp_k8s, "add_pod_label"):
        logger.warning(
            "Support for Pod labels is not yet available on the KFP 2 engine",
            project=function.metadata.project,
            function_name=function.metadata.name,
        )
        return

    kfp_k8s.add_pod_label(
        task, mlrun_constants.MLRunInternalLabels.mlrun_class, function.kind
    )
    kfp_k8s.add_pod_label(
        task, mlrun_constants.MLRunInternalLabels.function, function.metadata.name
    )
    kfp_k8s.add_pod_label(task, mlrun_constants.MLRunInternalLabels.name, task.name)
    kfp_k8s.add_pod_label(
        task, mlrun_constants.MLRunInternalLabels.project, function.metadata.project
    )
    kfp_k8s.add_pod_label(
        task, mlrun_constants.MLRunInternalLabels.tag, function.metadata.tag or "latest"
    )
    kfp_k8s.add_pod_label(
        task,
        mlrun_constants.MLRunInternalLabels.scrape_metrics,
        "True" if scrape_metrics else "False",
    )


def add_default_env(task):
    if hasattr(kfp_k8s, "use_field_path_as_env"):
        kfp_k8s.use_field_path_as_env(task, "MLRUN_NAMESPACE", "metadata.namespace")
    else:
        # TODO: remove this warning as soon as "use_field_path_as_env" is available for MLRun SDK
        logger.warning(
            "Support for field paths as Pod environment variables is not yet available for the KFP 2 engine."
            'Functions tentatively default to "MLRUN_NAMESPACE: mlrun"',
        )
        task.set_env_variable(name="MLRUN_NAMESPACE", value="mlrun")

    if config.httpdb.api_url:
        task.set_env_variable(name="MLRUN_DBPATH", value=config.httpdb.api_url)

    if config.mpijob_crd_version:
        task.set_env_variable(
            name="MLRUN_MPIJOB_CRD_VERSION", value=config.mpijob_crd_version
        )

    auth_env_var = (
        mlrun.common.runtimes.constants.FunctionEnvironmentVariables.auth_session
    )
    if auth_env_var in os.environ or "V3IO_ACCESS_KEY" in os.environ:
        task.set_env_variable(
            name=auth_env_var,
            value=os.environ.get(auth_env_var) or os.environ.get("V3IO_ACCESS_KEY"),
        )
    return task


def sync_environment_variables(function, task):
    function_env = {var.name: var.value for var in function.spec.env}
    for k in function_env:
        task.set_env_variable(name=k, value=function_env[k])
    return task


def sync_mounts(function, task):
    supported_mounts = {
        "configMap": __sync_mount_config_map,
        "secret": __sync_mount_secret,
        "PVC": __sync_pvc,
    }
    for volume in function.spec.volumes:
        for key in volume:
            if isinstance(volume[key], dict):
                mount_path = ""
                for m in function.spec.volume_mounts:
                    if m["name"] == volume["name"]:
                        mount_path = m["mountPath"]
                        break
                supported_mounts[key](task, volume, mount_path)
    return task


def __sync_mount_config_map(task, volume, mount_path):
    # TODO: remove this warning as soon as KFP SDK >=2.7.0 is available for MLRun SDK
    if not hasattr(kfp_k8s, "use_config_map_as_volume"):
        logger.warning(
            "Support for using a ConfigMap as a volume is not yet available on the KFP 2 engine",
        )
        return
    kfp_k8s.use_config_map_as_volume(task, volume["configMap"]["name"], mount_path)
    return task


def __sync_mount_secret(task, volume, mount_path):
    kfp_k8s.use_secret_as_volume(task, volume["secret"]["name"], mount_path)
    return task


def __sync_pvc(task, volume, mount_path):
    kfp_k8s.mount_pvc(task, volume["PVC"]["name"], mount_path)
    return task


def generate_pipeline_node(
    project_name: str,
    name: str,
    image: str,
    command: list,
    file_outputs: dict,
    function,
    func_url: str,
    scrape_metrics: bool,
    code_env: str,
    registry: str,
):
    def mlrun_function():
        return dsl.ContainerSpec(
            image=image,
            command=command,
        )

    container_component = dsl.component_factory.create_container_component_from_func(
        mlrun_function
    )

    task = container_component()
    task.set_display_name(name)

    add_default_function_resources(task, function)
    add_function_node_selection_attributes(function, task)
    add_annotations(task, PipelineRunType.run, function, func_url, project_name)
    add_labels(task, function, scrape_metrics)
    task.set_env_variable(
        name="MLRUN_ARTIFACT_PATH",
        value=mlrun.pipeline_context.project._enrich_artifact_path_with_workflow_uid(),
    )
    if code_env:
        task.set_env_variable(name="MLRUN_EXEC_CODE", value=code_env)
    if registry:
        task.set_env_variable(
            name="MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY", value=registry
        )
    add_default_env(task)
    sync_mounts(function, task)
    sync_environment_variables(function, task)
    return task


def generate_image_builder_pipeline_node(
    name,
    function=None,
    func_url=None,
    cmd=None,
):
    def build_mlrun_function(state: dsl.OutputPath(str), image: dsl.OutputPath(str)):
        runtime_args = ["--state-file-path", state, "--image-file-path", image]
        return dsl.ContainerSpec(
            image=config.kfp_image,
            command=cmd + runtime_args,
        )

    container_component = dsl.component_factory.create_container_component_from_func(
        build_mlrun_function
    )
    task = container_component()
    task.set_display_name(name)

    add_default_function_resources(task)
    add_function_node_selection_attributes(function, task)
    add_annotations(task, PipelineRunType.build, function, func_url)

    if config.httpdb.builder.docker_registry:
        task.set_env_variable(
            name="MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY",
            value=config.httpdb.builder.docker_registry,
        )
    if "IGZ_NAMESPACE_DOMAIN" in os.environ:
        task.set_env_variable(
            name="IGZ_NAMESPACE_DOMAIN",
            value=os.environ.get("IGZ_NAMESPACE_DOMAIN"),
        )

    is_v3io = function.spec.build.source and function.spec.build.source.startswith(
        "v3io"
    )
    if "V3IO_ACCESS_KEY" in os.environ and is_v3io:
        task.set_env_variable(
            name="V3IO_ACCESS_KEY", value=os.environ.get("V3IO_ACCESS_KEY")
        )
    add_default_env(task)
    return task


def generate_deployer_pipeline_node(
    name,
    function,
    func_url=None,
    cmd=None,
):
    def deploy_function():
        return dsl.ContainerSpec(
            image=config.kfp_image,
            command=cmd,
        )

    container_component = dsl.component_factory.create_container_component_from_func(
        deploy_function
    )
    task = container_component()
    task.set_display_name(name)

    add_default_function_resources(task)
    add_function_node_selection_attributes(function, task)
    add_annotations(task, PipelineRunType.deploy, function, func_url)

    add_default_env(task)
    return task
