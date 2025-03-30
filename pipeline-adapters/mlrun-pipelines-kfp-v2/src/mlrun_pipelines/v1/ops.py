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
import os.path

import inflection
from kubernetes import client as k8s_client

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
import mlrun.utils.helpers
import mlrun_pipelines.common.constants
import mlrun_pipelines.common.ops
from mlrun.config import config
from mlrun.utils import get_in
from mlrun_pipelines.common.helpers import (
    FUNCTION_ANNOTATION,
    PROJECT_ANNOTATION,
    RUN_ANNOTATION,
)
from mlrun_pipelines.common.ops import KFPMETA_DIR
from mlrun_pipelines.imports import dsl


def generate_deployer_pipeline_node(
    name,
    function,
    func_url=None,
    cmd=None,
):
    cop = dsl.ContainerOp(
        name=name,
        image=config.kfp_image,
        command=cmd,
        file_outputs={"endpoint": "/tmp/output", "name": "/tmp/name"},
    )
    cop = add_default_function_resources(container_op=cop, function=function)
    cop = add_function_node_selection_attributes(container_op=cop, function=function)

    add_annotations(
        cop, mlrun_pipelines.common.constants.PipelineRunType.deploy, function, func_url
    )
    add_default_env(k8s_client, cop)
    return cop


def add_env(env=None):
    """
    Modifier function to add env vars from dict
    Usage:
        train = train_op(...)
        train.apply(add_env({'MY_ENV':'123'}))
    """

    env = {} if env is None else env

    def _add_env(task):
        for k, v in env.items():
            task.add_env_variable(k8s_client.V1EnvVar(name=k, value=v))
        return task

    return _add_env


def generate_image_builder_pipeline_node(
    name,
    function=None,
    func_url=None,
    cmd=None,
):
    cop = dsl.ContainerOp(
        name=name,
        image=config.kfp_image,
        command=cmd,
        file_outputs={"state": "/tmp/state", "image": "/tmp/image"},
    )
    cop = add_default_function_resources(container_op=cop, function=function)
    cop = add_function_node_selection_attributes(container_op=cop, function=function)

    add_annotations(
        cop, mlrun_pipelines.common.constants.PipelineRunType.build, function, func_url
    )
    if config.httpdb.builder.docker_registry:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY",
                value=config.httpdb.builder.docker_registry,
            )
        )
    if "IGZ_NAMESPACE_DOMAIN" in os.environ:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="IGZ_NAMESPACE_DOMAIN",
                value=os.environ.get("IGZ_NAMESPACE_DOMAIN"),
            )
        )

    is_v3io = function.spec.build.source and function.spec.build.source.startswith(
        "v3io"
    )
    if "V3IO_ACCESS_KEY" in os.environ and is_v3io:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="V3IO_ACCESS_KEY", value=os.environ.get("V3IO_ACCESS_KEY")
            )
        )

    add_default_env(k8s_client, cop)

    return cop


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
    cop = dsl.ContainerOp(
        name=name,
        image=image,
        command=command,
        file_outputs=file_outputs,
        output_artifact_paths={
            "mlpipeline-ui-metadata": os.path.join(
                KFPMETA_DIR, "mlpipeline-ui-metadata.json"
            ),
            "mlpipeline-metrics": os.path.join(KFPMETA_DIR, "mlpipeline-metrics.json"),
        },
    )
    cop = add_default_function_resources(container_op=cop, function=function)
    cop = add_function_node_selection_attributes(container_op=cop, function=function)

    add_annotations(
        cop,
        mlrun_pipelines.common.constants.PipelineRunType.run,
        function,
        func_url,
        project_name,
    )
    add_labels(cop, function, scrape_metrics)
    if code_env:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_EXEC_CODE", value=code_env)
        )
    if registry:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY", value=registry
            )
        )

    add_default_env(k8s_client, cop)

    return cop


def add_default_env(k8s_client, cop):
    cop.container.add_env_variable(
        k8s_client.V1EnvVar(
            "MLRUN_NAMESPACE",
            value_from=k8s_client.V1EnvVarSource(
                field_ref=k8s_client.V1ObjectFieldSelector(
                    field_path="metadata.namespace"
                )
            ),
        )
    )

    if config.httpdb.api_url:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_DBPATH", value=config.httpdb.api_url)
        )

    if config.mpijob_crd_version:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="MLRUN_MPIJOB_CRD_VERSION", value=config.mpijob_crd_version
            )
        )

    auth_env_var = (
        mlrun.common.runtimes.constants.FunctionEnvironmentVariables.auth_session
    )
    if auth_env_var in os.environ or "V3IO_ACCESS_KEY" in os.environ:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name=auth_env_var,
                value=os.environ.get(auth_env_var) or os.environ.get("V3IO_ACCESS_KEY"),
            )
        )


def add_annotations(cop, kind, function, func_url=None, project=None):
    if func_url and func_url.startswith("db://"):
        func_url = func_url[len("db://") :]
    cop.add_pod_annotation(RUN_ANNOTATION, kind)
    cop.add_pod_annotation(PROJECT_ANNOTATION, project or function.metadata.project)
    cop.add_pod_annotation(FUNCTION_ANNOTATION, func_url or function.uri)


def add_labels(cop, function, scrape_metrics=False):
    cop.add_pod_label(mlrun_constants.MLRunInternalLabels.mlrun_class, function.kind)
    cop.add_pod_label(
        mlrun_constants.MLRunInternalLabels.function, function.metadata.name
    )
    cop.add_pod_label(mlrun_constants.MLRunInternalLabels.name, cop.human_name)
    cop.add_pod_label(
        mlrun_constants.MLRunInternalLabels.project, function.metadata.project
    )
    cop.add_pod_label(
        mlrun_constants.MLRunInternalLabels.tag, function.metadata.tag or "latest"
    )
    cop.add_pod_label(
        mlrun_constants.MLRunInternalLabels.scrape_metrics,
        "True" if scrape_metrics else "False",
    )


def add_default_function_resources(
    container_op,
    function: dsl.ContainerOp,
) -> dsl.ContainerOp:
    default_resources = config.get_default_function_pod_resources()
    for resource_name, resource_value in default_resources["requests"].items():
        if resource_value:
            container_op.container.add_resource_request(resource_name, resource_value)

    for resource_name, resource_value in default_resources["limits"].items():
        if resource_value:
            container_op.container.add_resource_limit(resource_name, resource_value)
    mlrun_pipelines.common.ops._enrich_gpu_limits(function=function, task=container_op)
    return container_op


def add_function_node_selection_attributes(
    function, container_op: dsl.ContainerOp
) -> dsl.ContainerOp:
    if not mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind):
        enriched_node_selector = mlrun_pipelines.common.ops._enrich_node_selector(
            function
        )
        if enriched_node_selector:
            container_op.node_selector = enriched_node_selector

        if getattr(function.spec, "tolerations"):
            container_op.tolerations = function.spec.tolerations

        if getattr(function.spec, "affinity"):
            container_op.affinity = function.spec.affinity

    return container_op


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.workflow_manifest()
    if not workflow:
        return None, project, None

    templates = {}
    for template in workflow["spec"]["templates"]:
        project = project or get_in(
            template, ["metadata", "annotations", PROJECT_ANNOTATION], ""
        )
        name = template["name"]
        templates[name] = {
            "run_type": get_in(
                template, ["metadata", "annotations", RUN_ANNOTATION], ""
            ),
            "function": get_in(
                template, ["metadata", "annotations", FUNCTION_ANNOTATION], ""
            ),
        }

    nodes = workflow["status"].get("nodes", {})
    dag = {}
    for node in nodes.values():
        name = node["displayName"]
        record = {
            k: node[k] for k in ["phase", "startedAt", "finishedAt", "type", "id"]
        }

        # snake case
        # align kfp fields to mlrun snake case convention
        # create snake_case for consistency.
        # retain the camelCase for compatibility
        for key in list(record.keys()):
            record[inflection.underscore(key)] = record[key]

        record["parent"] = node.get("boundaryID", "")
        record["name"] = name
        record["children"] = node.get("children", [])
        if name in templates:
            record["function"] = templates[name].get("function")
            record["run_type"] = templates[name].get("run_type")
        dag[node["id"]] = record

    return dag, project, workflow["status"].get("message", "")
