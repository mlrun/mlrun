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
#

import os

from kfp import dsl
from kfp import kubernetes as kfp_k8s

import mlrun
from mlrun.config import config
from mlrun.pipelines.common.helpers import (
    function_annotation,
    project_annotation,
    run_annotation,
)
from mlrun.pipelines.common.ops import PipelineRunType
from mlrun.utils import get_in, logger


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.workflow_manifest
    if not workflow:
        return None, project, None

    # TODO: this method does not take into account that KFP API 2.0 might return a manifest with the 1.8 schema

    templates = {}
    for name, template in workflow.get_executors():
        project = project or get_in(
            template, ["metadata", "annotations", project_annotation], ""
        )
        templates[name] = {
            "run_type": get_in(
                template, ["metadata", "annotations", run_annotation], ""
            ),
            "function": get_in(
                template, ["metadata", "annotations", function_annotation], ""
            ),
        }

    dag = {}
    nodes = []
    if run["run_details"]:
        nodes = run["run_details"].get("task_details", [])
    for node in nodes:
        name = node["display_name"]
        record = {
            "phase": node["state"],
            "started_at": node["create_time"],
            "finished_at": node["end_time"],
            "id": node["display_name"],
            "parent": node.get("display_name", ""),
            "name": name,
            "type": "DAG" if node["child_tasks"] else "Pod",
            "children": [c["pod_name"] for c in node["child_tasks"] or []],
        }

        if name in templates:
            record["function"] = templates[name].get("function")
            record["run_type"] = templates[name].get("run_type")
        dag[node["display_name"]] = record

    # TODO: find workflow exit message on the KFP 2.0 API and use it instead of "state"
    return dag, project, run["state"]


def add_default_function_resources(
    task: dsl.PipelineTask,
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

    return task


def add_function_node_selection_attributes(
    function, task: dsl.PipelineTask
) -> dsl.PipelineTask:
    if not mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind):
        if getattr(function.spec, "node_selector"):
            for k, v in function.spec.node_selector.items():
                task = kfp_k8s.add_node_selector(task, k, v)

    # TODO: is there a way to also set tolerations and affinities to a task?

    return task


def add_annotations(
    task: dsl.PipelineTask,
    kind: str,
    function,
    func_url: str = None,
    project: str = None,
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
    kfp_k8s.add_pod_annotation(task, run_annotation, kind)
    kfp_k8s.add_pod_annotation(
        task, project_annotation, project or function.metadata.project
    )
    kfp_k8s.add_pod_annotation(task, function_annotation, func_url or function.uri)


def add_labels(task, function, scrape_metrics=False):
    # TODO: remove this warning as soon as KFP SDK >=2.7.0 is available for MLRun SDK
    if not hasattr(kfp_k8s, "add_pod_label"):
        logger.warning(
            "Support for Pod labels is not yet available on the KFP 2 engine",
            project=function.metadata.project,
            function_name=function.metadata.name,
        )
        return

    prefix = mlrun.runtimes.utils.mlrun_key
    kfp_k8s.add_pod_label(task, prefix + "class", function.kind)
    kfp_k8s.add_pod_label(task, prefix + "function", function.metadata.name)
    kfp_k8s.add_pod_label(task, prefix + "name", task.name)
    kfp_k8s.add_pod_label(task, prefix + "project", function.metadata.project)
    kfp_k8s.add_pod_label(task, prefix + "tag", function.metadata.tag or "latest")
    kfp_k8s.add_pod_label(
        task, prefix + "scrape-metrics", "True" if scrape_metrics else "False"
    )


def add_default_env(task):
    # TODO: how to add an env variable that references the metadata namespace attribute?
    #  "MLRUN_NAMESPACE" depends on it
    task.set_env_variable(name="MLRUN_NAMESPACE", value="mlrun")

    if config.httpdb.api_url:
        task.set_env_variable(name="MLRUN_DBPATH", value=config.httpdb.api_url)

    if config.mpijob_crd_version:
        task.set_env_variable(
            name="MLRUN_MPIJOB_CRD_VERSION", value=config.mpijob_crd_version
        )

    auth_env_var = mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session
    if auth_env_var in os.environ or "V3IO_ACCESS_KEY" in os.environ:
        task.set_env_variable(
            name=auth_env_var,
            value=os.environ.get(auth_env_var) or os.environ.get("V3IO_ACCESS_KEY"),
        )


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
    csp = dsl.ContainerSpec(
        image=image,
        command=command,
    )

    container_component = dsl.component_factory.create_container_component_from_func(
        lambda: csp
    )
    task = container_component()

    # TODO: work out how to reproduce the file and artifacts output behaviour used on ContainerOp

    # TODO: ensure that "name" can be used to identify the current pipeline node
    task.set_display_name(name)

    task = add_default_function_resources(task)
    task = add_function_node_selection_attributes(function, task)

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
    return task


def build_op(
    name,
    function=None,
    func_url=None,
    image=None,
    base_image=None,
    commands: list = None,
    secret_name="",
    with_mlrun=True,
    skip_deployed=False,
):
    """build Docker image."""
    raise NotImplementedError


def deploy_op(
    name,
    function,
    func_url=None,
    source="",
    project="",
    models: list = None,
    env: dict = None,
    tag="",
    verbose=False,
):
    raise NotImplementedError
