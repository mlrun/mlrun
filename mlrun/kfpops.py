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

import getpass
import json
import os
import os.path
from copy import deepcopy

import mlrun

from .config import config
from .db import get_or_set_dburl, get_run_db
from .model import HyperParamOptions
from .utils import (
    dict_to_yaml,
    gen_md_table,
    get_artifact_target,
    get_in,
    get_workflow_url,
    is_ipython,
    logger,
    run_keys,
)

KFPMETA_DIR = os.environ.get("KFPMETA_OUT_DIR", "")
KFP_ARTIFACTS_DIR = os.environ.get("KFP_ARTIFACTS_DIR", "/tmp")

project_annotation = "mlrun/project"
run_annotation = "mlrun/pipeline-step-type"
function_annotation = "mlrun/function-uri"


class PipelineRunType:
    run = "run"
    build = "build"
    deploy = "deploy"


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

    results["run_id"] = results.get("run_id", "/".join([project, uid]))
    for key in struct["spec"].get(run_keys.outputs, []):
        val = "None"
        if key in out_dict:
            val = out_dict[key]
        elif key in results:
            val = results[key]
        try:
            path = "/".join([KFP_ARTIFACTS_DIR, key])
            logger.info("writing artifact output", path=path, val=val)
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
    with open(KFPMETA_DIR + "/mlpipeline-ui-metadata.json", "w") as f:
        json.dump(metadata, f)


def get_kfp_outputs(artifacts, labels, project):
    outputs = []
    out_dict = {}
    for output in artifacts:
        key = output["key"]
        target = output.get("target_path", "")
        target = output.get("inline", target)
        out_dict[key] = get_artifact_target(output, project=project)

        if target.startswith("v3io:///"):
            target = target.replace("v3io:///", "http://v3io-webapi:8081/")

        user = labels.get("v3io_user", "") or os.environ.get("V3IO_USERNAME", "")
        if target.startswith("/User/"):
            user = user or "admin"
            target = "http://v3io-webapi:8081/users/" + user + target[5:]

        viewer = output.get("viewer", "")
        if viewer in ["web-app", "chart"]:
            meta = {"type": "web-app", "source": target}
            outputs += [meta]

        elif viewer == "table":
            header = output.get("header", None)
            if header and target.endswith(".csv"):
                meta = {
                    "type": "table",
                    "format": "csv",
                    "header": header,
                    "source": target,
                }
                outputs += [meta]

        elif output["kind"] == "dataset":
            header = output.get("header")
            preview = output.get("preview")
            if preview:
                tbl_md = gen_md_table(header, preview)
                text = f"## Dataset: {key}  \n\n" + tbl_md
                del output["preview"]

                meta = {"type": "markdown", "storage": "inline", "source": text}
                outputs += [meta]

    return outputs, out_dict


def mlrun_op(
    name: str = "",
    project: str = "",
    function=None,
    func_url=None,
    image: str = "",
    runobj=None,
    command: str = "",
    secrets: list = None,
    params: dict = None,
    job_image=None,
    hyperparams: dict = None,
    param_file: str = "",
    labels: dict = None,
    selector: str = "",
    inputs: dict = None,
    outputs: list = None,
    in_path: str = "",
    out_path: str = "",
    rundb: str = "",
    mode: str = "",
    handler: str = "",
    more_args: list = None,
    hyper_param_options=None,
    verbose=None,
    scrape_metrics=False,
):
    """mlrun KubeFlow pipelines operator, use to form pipeline steps

    when using kubeflow pipelines, each step is wrapped in an mlrun_op
    one step can pass state and data to the next step, see example below.

    :param name:    name used for the step
    :param project: optional, project name
    :param image:   optional, run container image (will be executing the step)
                    the container should host all required packages + code
                    for the run, alternatively user can mount packages/code via
                    shared file volumes like v3io (see example below)
    :param function: optional, function object
    :param func_url: optional, function object url
    :param command: exec command (or URL for functions)
    :param secrets: extra secrets specs, will be injected into the runtime
                    e.g. ['file=<filename>', 'env=ENV_KEY1,ENV_KEY2']
    :param params:  dictionary of run parameters and values
    :param hyperparams: dictionary of hyper parameters and list values, each
                        hyperparam holds a list of values, the run will be
                        executed for every parameter combination (GridSearch)
    :param param_file:  a csv/json file with parameter combinations, first csv row hold
                        the parameter names, following rows hold param values
    :param selector: selection criteria for hyperparams e.g. "max.accuracy"
    :param hyper_param_options: hyper param options class, see: :py:class:`~mlrun.model.HyperParamOptions`
    :param labels:   labels to tag the job/run with ({key:val, ..})
    :param inputs:   dictionary of input objects + optional paths (if path is
                     omitted the path will be the in_path/key.
    :param outputs:  dictionary of input objects + optional paths (if path is
                     omitted the path will be the out_path/key.
    :param in_path:  default input path/url (prefix) for inputs
    :param out_path: default output path/url (prefix) for artifacts
    :param rundb:    path for rundb (or use 'MLRUN_DBPATH' env instead)
    :param mode:     run mode, e.g. 'pass' for using the command without mlrun wrapper
    :param handler   code entry-point/hanfler name
    :param job_image name of the image user for the job
    :param verbose:  add verbose prints/logs
    :param scrape_metrics:  whether to add the `mlrun/scrape-metrics` label to this run's resources

    :returns: KFP step operation

    Example:
    from kfp import dsl
    from mlrun import mlrun_op
    from mlrun.platforms import mount_v3io

    def mlrun_train(p1, p2):
    return mlrun_op('training',
                    command = '/User/kubeflow/training.py',
                    params = {'p1':p1, 'p2':p2},
                    outputs = {'model.txt':'', 'dataset.csv':''},
                    out_path ='v3io:///projects/my-proj/mlrun/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    # use data from the first step
    def mlrun_validate(modelfile):
        return mlrun_op('validation',
                    command = '/User/kubeflow/validation.py',
                    inputs = {'model.txt':modelfile},
                    out_path ='v3io:///projects/my-proj/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    @dsl.pipeline(
        name='My MLRUN pipeline', description='Shows how to use mlrun.'
    )
    def mlrun_pipeline(
        p1 = 5 , p2 = '"text"'
    ):
        # run training, mount_v3io will mount "/User" into the pipeline step
        train = mlrun_train(p1, p2).apply(mount_v3io())

        # feed 1st step results into the secound step
        validate = mlrun_validate(
            train.outputs['model-txt']).apply(mount_v3io())

    """
    from os import environ

    from kfp import dsl
    from kubernetes import client as k8s_client

    secrets = [] if secrets is None else secrets
    params = {} if params is None else params
    hyperparams = {} if hyperparams is None else hyperparams
    if hyper_param_options and isinstance(hyper_param_options, dict):
        hyper_param_options = HyperParamOptions.from_dict(hyper_param_options)
    inputs = {} if inputs is None else inputs
    outputs = [] if outputs is None else outputs
    labels = {} if labels is None else labels

    rundb = rundb or get_or_set_dburl()
    cmd = [
        "python",
        "-m",
        "mlrun",
        "run",
        "--kfp",
        "--from-env",
        "--workflow",
        "{{workflow.uid}}",
    ]
    file_outputs = {}

    runtime = None
    code_env = None
    function_name = ""
    if function:

        if not func_url:
            if function.kind in ["", "local"]:
                image = image or function.spec.image
                command = command or function.spec.command
                more_args = more_args or function.spec.args
                mode = mode or function.spec.mode
                rundb = rundb or function.spec.rundb
                code_env = str(function.spec.build.functionSourceCode)
            else:
                runtime = str(function.to_dict())

        function_name = function.metadata.name
        if function.kind == "dask":
            image = image or function.spec.kfp_image or config.dask_kfp_image

    image = image or config.kfp_image

    if runobj:
        handler = handler or runobj.spec.handler_name
        params = params or runobj.spec.parameters
        hyperparams = hyperparams or runobj.spec.hyperparams
        param_file = (
            param_file
            or runobj.spec.param_file
            or runobj.spec.hyper_param_options.param_file
        )
        hyper_param_options = hyper_param_options or runobj.spec.hyper_param_options
        selector = (
            selector or runobj.spec.selector or runobj.spec.hyper_param_options.selector
        )
        inputs = inputs or runobj.spec.inputs
        outputs = outputs or runobj.spec.outputs
        in_path = in_path or runobj.spec.input_path
        out_path = out_path or runobj.spec.output_path
        secrets = secrets or runobj.spec.secret_sources
        project = project or runobj.metadata.project
        labels = runobj.metadata.labels or labels
        verbose = verbose or runobj.spec.verbose
        scrape_metrics = scrape_metrics or runobj.spec.scrape_metrics

    if not name:
        if not function_name:
            raise ValueError("name or function object must be specified")
        name = function_name
        if handler:
            name += "-" + handler

    if hyperparams or param_file:
        outputs.append("iteration_results")
    if "run_id" not in outputs:
        outputs.append("run_id")

    params = params or {}
    hyperparams = hyperparams or {}
    inputs = inputs or {}
    secrets = secrets or []

    if "V3IO_USERNAME" in environ and "v3io_user" not in labels:
        labels["v3io_user"] = os.environ.get("V3IO_USERNAME")
    if "owner" not in labels:
        labels["owner"] = os.environ.get("V3IO_USERNAME") or getpass.getuser()

    if name:
        cmd += ["--name", name]
    if func_url:
        cmd += ["-f", func_url]
    for secret in secrets:
        cmd += ["-s", f"{secret['kind']}={secret['source']}"]
    for param, val in params.items():
        cmd += ["-p", f"{param}={val}"]
    for xpram, val in hyperparams.items():
        cmd += ["-x", f"{xpram}={val}"]
    for input_param, val in inputs.items():
        cmd += ["-i", f"{input_param}={val}"]
    for label, val in labels.items():
        cmd += ["--label", f"{label}={val}"]
    for output in outputs:
        cmd += ["-o", str(output)]
        file_outputs[
            output.replace(".", "_")
        ] = f"/tmp/{output}"  # not using path.join to avoid windows "\"
    if project:
        cmd += ["--project", project]
    if handler:
        cmd += ["--handler", handler]
    if runtime:
        cmd += ["--runtime", runtime]
    if in_path:
        cmd += ["--in-path", in_path]
    if out_path:
        cmd += ["--out-path", out_path]
    if param_file:
        cmd += ["--param-file", param_file]
    if hyper_param_options:
        cmd += ["--hyper-param-options", hyper_param_options.to_json()]
    if selector:
        cmd += ["--selector", selector]
    if job_image:
        cmd += ["--image", job_image]
    if mode:
        cmd += ["--mode", mode]
    if verbose:
        cmd += ["--verbose"]
    if scrape_metrics:
        cmd += ["--scrape-metrics"]
    if more_args:
        cmd += more_args

    registry = get_default_reg()
    if image and image.startswith("."):
        if registry:
            image = f"{registry}/{image[1:]}"
        else:
            raise ValueError("local image registry env not found")

    cop = dsl.ContainerOp(
        name=name,
        image=image,
        command=cmd + [command],
        file_outputs=file_outputs,
        output_artifact_paths={
            "mlpipeline-ui-metadata": "/mlpipeline-ui-metadata.json",
            "mlpipeline-metrics": "/mlpipeline-metrics.json",
        },
    )

    add_annotations(cop, PipelineRunType.run, function, func_url, project)
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


def deploy_op(
    name,
    function,
    func_url=None,
    source="",
    dashboard="",
    project="",
    models: list = None,
    env: dict = None,
    tag="",
    verbose=False,
):
    from kfp import dsl
    from kubernetes import client as k8s_client

    cmd = ["python", "-m", "mlrun", "deploy"]
    if source:
        cmd += ["-s", source]
    if dashboard:
        cmd += ["-d", dashboard]
    if tag:
        cmd += ["--tag", tag]
    if verbose:
        cmd += ["--verbose"]
    if project:
        cmd += ["-p", project]

    if models:
        for m in models:
            for key in ["model_path", "model_url", "class_name"]:
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
        from kubernetes import client as k8s_client

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

    from os import environ

    from kfp import dsl
    from kubernetes import client as k8s_client

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
        cmd += ["--with_mlrun"]
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

    add_annotations(cop, PipelineRunType.build, function, func_url)
    if config.httpdb.builder.docker_registry:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY",
                value=config.httpdb.builder.docker_registry,
            )
        )
    if "IGZ_NAMESPACE_DOMAIN" in environ:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="IGZ_NAMESPACE_DOMAIN",
                value=os.environ.get("IGZ_NAMESPACE_DOMAIN"),
            )
        )

    is_v3io = function.spec.build.source and function.spec.build.source.startswith(
        "v3io"
    )
    if "V3IO_ACCESS_KEY" in environ and is_v3io:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="V3IO_ACCESS_KEY", value=os.environ.get("V3IO_ACCESS_KEY")
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

    if "MLRUN_AUTH_SESSION" in os.environ or "V3IO_ACCESS_KEY" in os.environ:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="MLRUN_AUTH_SESSION",
                value=os.environ.get("MLRUN_AUTH_SESSION")
                or os.environ.get("V3IO_ACCESS_KEY"),
            )
        )


def get_default_reg():
    if config.httpdb.builder.docker_registry:
        return config.httpdb.builder.docker_registry
    namespace_domain = os.environ.get("IGZ_NAMESPACE_DOMAIN", None)
    if namespace_domain is not None:
        return f"docker-registry.{namespace_domain}:80"
    return ""


def add_annotations(cop, kind, function, func_url=None, project=None):
    if func_url and func_url.startswith("db://"):
        func_url = func_url[len("db://") :]
    cop.add_pod_annotation(run_annotation, kind)
    cop.add_pod_annotation(project_annotation, project or function.metadata.project)
    cop.add_pod_annotation(function_annotation, func_url or function.uri)


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.get("pipeline_runtime", {}).get("workflow_manifest")
    if not workflow:
        return None, project, None
    workflow = json.loads(workflow)

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
        record["parent"] = node.get("boundaryID", "")
        record["name"] = name
        record["children"] = node.get("children", [])
        if name in templates:
            record["function"] = templates[name].get("function")
            record["run_type"] = templates[name].get("run_type")
        dag[node["id"]] = record

    return dag, project, workflow["status"].get("message", "")


def format_summary_from_kfp_run(kfp_run, project=None, session=None):
    override_project = project if project and project != "*" else None
    dag, project, message = generate_kfp_dag_and_resolve_project(
        kfp_run, override_project
    )
    run_id = get_in(kfp_run, "run.id")

    # enrich DAG with mlrun run info
    if session:
        runs = mlrun.api.utils.singletons.db.get_db().list_runs(
            session, project=project, labels=f"workflow={run_id}"
        )
    else:
        runs = get_run_db().list_runs(project=project, labels=f"workflow={run_id}")

    for run in runs:
        step = get_in(run, ["metadata", "labels", "mlrun/runner-pod"])
        if step and step in dag:
            dag[step]["run_uid"] = get_in(run, "metadata.uid")
            dag[step]["kind"] = get_in(run, "metadata.labels.kind")
            error = get_in(run, "status.error")
            if error:
                dag[step]["error"] = error

    short_run = {"graph": dag}
    short_run["run"] = {
        k: str(v)
        for k, v in kfp_run["run"].items()
        if k
        in [
            "id",
            "name",
            "status",
            "error",
            "created_at",
            "scheduled_at",
            "finished_at",
            "description",
        ]
    }
    short_run["run"]["project"] = project
    short_run["run"]["message"] = message
    return short_run


def show_kfp_run(run, clear_output=False):
    phase_to_color = {
        mlrun.run.RunStatuses.failed: "red",
        mlrun.run.RunStatuses.succeeded: "green",
        mlrun.run.RunStatuses.skipped: "white",
    }
    runtype_to_shape = {
        PipelineRunType.run: "ellipse",
        PipelineRunType.build: "box",
        PipelineRunType.deploy: "box3d",
    }
    if not run or "graph" not in run:
        return
    if is_ipython:
        try:
            from graphviz import Digraph
        except ImportError:
            return

        try:
            graph = run["graph"]
            dag = Digraph("kfp", format="svg")
            dag.attr(compound="true")

            for key, node in graph.items():
                if node["type"] != "DAG" or node["parent"]:
                    shape = "ellipse"
                    if node.get("run_type"):
                        shape = runtype_to_shape.get(node["run_type"], None)
                    elif node["phase"] == "Skipped" or (
                        node["type"] == "DAG" and node["name"].startswith("condition-")
                    ):
                        shape = "diamond"
                    dag.node(
                        key,
                        label=node["name"],
                        fillcolor=phase_to_color.get(node["phase"], None),
                        style="filled",
                        shape=shape,
                        tooltip=node.get("error", None),
                    )
                    for child in node.get("children") or []:
                        dag.edge(key, child)

            import IPython

            if clear_output:
                IPython.display.clear_output(wait=True)

            run_id = run["run"]["id"]
            url = get_workflow_url(run["run"]["project"], run_id)
            href = f'<a href="{url}" target="_blank"><b>click here</b></a>'
            html = IPython.display.HTML(
                f"<div>Pipeline running (id={run_id}), {href} to view the details in MLRun UI</div>"
            )
            IPython.display.display(html, dag)
        except Exception as exc:
            logger.warning(f"failed to plot graph, {exc}")
