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

import json
import os
import os.path
from copy import deepcopy

import inflection
from kfp import dsl
from kubernetes import client as k8s_client
from mlrun_pipelines.common.helpers import (
    function_annotation,
    project_annotation,
    run_annotation,
)
from mlrun_pipelines.common.ops import KFP_ARTIFACTS_DIR, KFPMETA_DIR, PipelineRunType

import mlrun
from mlrun.config import config
from mlrun.utils import (
    dict_to_yaml,
    gen_md_table,
    get_artifact_target,
    get_in,
    is_legacy_artifact,
    logger,
    run_keys,
)

dsl.ContainerOp._DISABLE_REUSABLE_COMPONENT_WARNING = True


def is_num(v):
    return isinstance(v, (int, float, complex))


def write_kfpmeta(struct):
    if "status" not in struct:
        return

    results = struct["status"].get("results", {})
    metrics = {
        "metrics": [
            {"name": k, "numberValue": v} for k, v in results.items() if is_num(v)
        ],
    }
    with open(KFPMETA_DIR + "/mlpipeline-metrics.json", "w") as f:
        json.dump(metrics, f)

    struct = deepcopy(struct)
    uid = struct["metadata"].get("uid")
    project = struct["metadata"].get("project", config.default_project)
    output_artifacts, out_dict = get_kfp_outputs(
        struct["status"].get(run_keys.artifacts, []),
        struct["metadata"].get("labels", {}),
        project,
    )

    # /tmp/run_id
    results["run_id"] = results.get("run_id", "/".join([project, uid]))
    for key in struct["spec"].get(run_keys.outputs, []):
        val = "None"
        if key in out_dict:
            val = out_dict[key]
        elif key in results:
            val = results[key]
        try:
            path = "/".join([KFP_ARTIFACTS_DIR, key])
            logger.info("Writing artifact output", path=path, val=val)
            with open(path, "w") as fp:
                fp.write(str(val))
        except Exception as exc:
            logger.warning("Failed writing to temp file. Ignoring", exc=repr(exc))
            pass

    text = "# Run Report\n"
    if "iterations" in struct["status"]:
        del struct["status"]["iterations"]

    text += "## Metadata\n```yaml\n" + dict_to_yaml(struct) + "```\n"

    metadata = {
        "outputs": output_artifacts
        + [{"type": "markdown", "storage": "inline", "source": text}]
    }

    # saar is working on removing this
    with open(KFPMETA_DIR + "/mlpipeline-ui-metadata.json", "w") as f:
        json.dump(metadata, f)


def get_kfp_outputs(artifacts, labels, project):
    outputs = []
    out_dict = {}
    for output in artifacts:
        if is_legacy_artifact(output):
            key = output["key"]
            # The spec in a legacy artifact is contained in the main object, so using this assignment saves us a lot
            # of if/else in the rest of this function.
            output_spec = output
        else:
            key = output.get("metadata")["key"]
            output_spec = output.get("spec", {})

        target = output_spec.get("target_path", "")
        target = output_spec.get("inline", target)

        out_dict[key] = get_artifact_target(output, project=project)

        if target.startswith("v3io:///"):
            target = target.replace("v3io:///", "http://v3io-webapi:8081/")

        user = labels.get("v3io_user", "") or os.environ.get("V3IO_USERNAME", "")
        if target.startswith("/User/"):
            user = user or "admin"
            target = "http://v3io-webapi:8081/users/" + user + target[5:]

        viewer = output_spec.get("viewer", "")
        if viewer in ["web-app", "chart"]:
            meta = {"type": "web-app", "source": target}
            outputs += [meta]

        elif viewer == "table":
            header = output_spec.get("header", None)
            if header and target.endswith(".csv"):
                meta = {
                    "type": "table",
                    "format": "csv",
                    "header": header,
                    "source": target,
                }
                outputs += [meta]

        elif output.get("kind") == "dataset":
            header = output_spec.get("header")
            preview = output_spec.get("preview")
            if preview:
                tbl_md = gen_md_table(header, preview)
                text = f"## Dataset: {key}  \n\n" + tbl_md
                del output_spec["preview"]

                meta = {"type": "markdown", "storage": "inline", "source": text}
                outputs += [meta]

    return outputs, out_dict


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
    cmd = ["python", "-m", "mlrun", "deploy"]
    if source:
        cmd += ["-s", source]
    if tag:
        cmd += ["--tag", tag]
    if verbose:
        cmd += ["--verbose"]
    if project:
        cmd += ["-p", project]

    if models:
        for m in models:
            for key in ["key", "model_path", "model_url", "class_name", "model_url"]:
                if key in m:
                    m[key] = str(m[key])  # verify we stringify pipeline params
            if function.kind == mlrun.runtimes.RuntimeKinds.serving:
                cmd += ["-m", json.dumps(m)]
            else:
                cmd += ["-m", f"{m['key']}={m['model_path']}"]

    if env:
        for key, val in env.items():
            cmd += ["--env", f"{key}={val}"]

    if func_url:
        cmd += ["-f", func_url]
    else:
        runtime = f"{function.to_dict()}"
        cmd += [runtime]

    cop = dsl.ContainerOp(
        name=name,
        image=config.kfp_image,
        command=cmd,
        file_outputs={"endpoint": "/tmp/output", "name": "/tmp/name"},
    )
    cop = add_default_function_resources(cop)
    cop = add_function_node_selection_attributes(container_op=cop, function=function)

    add_annotations(cop, PipelineRunType.deploy, function, func_url)
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

    cmd = ["python", "-m", "mlrun", "build", "--kfp"]
    if function:
        if not hasattr(function, "to_dict"):
            raise ValueError("function must specify a function runtime object")
        cmd += ["-r", str(function.to_dict())]
    elif not func_url:
        raise ValueError("function object or func_url must be specified")

    commands = commands or []
    if image:
        cmd += ["-i", image]
    if base_image:
        cmd += ["-b", base_image]
    if secret_name:
        cmd += ["--secret-name", secret_name]
    if with_mlrun:
        cmd += ["--with-mlrun"]
    if skip_deployed:
        cmd += ["--skip"]
    for c in commands:
        cmd += ["-c", c]
    if func_url and not function:
        cmd += [func_url]

    cop = dsl.ContainerOp(
        name=name,
        image=config.kfp_image,
        command=cmd,
        file_outputs={"state": "/tmp/state", "image": "/tmp/image"},
    )
    cop = add_default_function_resources(cop)
    cop = add_function_node_selection_attributes(container_op=cop, function=function)

    add_annotations(cop, PipelineRunType.build, function, func_url)
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
            "mlpipeline-ui-metadata": KFPMETA_DIR + "/mlpipeline-ui-metadata.json",
            "mlpipeline-metrics": KFPMETA_DIR + "/mlpipeline-metrics.json",
        },
    )
    cop = add_default_function_resources(cop)
    cop = add_function_node_selection_attributes(container_op=cop, function=function)

    add_annotations(cop, PipelineRunType.run, function, func_url, project_name)
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

    auth_env_var = mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session
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
    cop.add_pod_annotation(run_annotation, kind)
    cop.add_pod_annotation(project_annotation, project or function.metadata.project)
    cop.add_pod_annotation(function_annotation, func_url or function.uri)


def add_labels(cop, function, scrape_metrics=False):
    prefix = mlrun.runtimes.utils.mlrun_key
    cop.add_pod_label(prefix + "class", function.kind)
    cop.add_pod_label(prefix + "function", function.metadata.name)
    cop.add_pod_label(prefix + "name", cop.human_name)
    cop.add_pod_label(prefix + "project", function.metadata.project)
    cop.add_pod_label(prefix + "tag", function.metadata.tag or "latest")
    cop.add_pod_label(prefix + "scrape-metrics", "True" if scrape_metrics else "False")


def add_default_function_resources(
    container_op: dsl.ContainerOp,
) -> dsl.ContainerOp:
    default_resources = config.get_default_function_pod_resources()
    for resource_name, resource_value in default_resources["requests"].items():
        if resource_value:
            container_op.container.add_resource_request(resource_name, resource_value)

    for resource_name, resource_value in default_resources["limits"].items():
        if resource_value:
            container_op.container.add_resource_limit(resource_name, resource_value)
    return container_op


def add_function_node_selection_attributes(
    function, container_op: dsl.ContainerOp
) -> dsl.ContainerOp:
    if not mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind):
        if getattr(function.spec, "node_selector"):
            container_op.node_selector = function.spec.node_selector

        if getattr(function.spec, "tolerations"):
            container_op.tolerations = function.spec.tolerations

        if getattr(function.spec, "affinity"):
            container_op.affinity = function.spec.affinity

    return container_op


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.workflow_manifest
    if not workflow:
        return None, project, None

    templates = {}
    for template in workflow["spec"]["templates"]:
        project = project or get_in(
            template, ["metadata", "annotations", project_annotation], ""
        )
        name = template["name"]
        templates[name] = {
            "run_type": get_in(
                template, ["metadata", "annotations", run_annotation], ""
            ),
            "function": get_in(
                template, ["metadata", "annotations", function_annotation], ""
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
