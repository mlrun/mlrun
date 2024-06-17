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
from copy import deepcopy
from typing import Union

import mlrun_pipelines.common.models

import mlrun
import mlrun.common.constants
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.model import HyperParamOptions, RunSpec
from mlrun.utils import (
    dict_to_yaml,
    gen_md_table,
    get_artifact_target,
    get_in,
    get_workflow_url,
    is_ipython,
    logger,
    run_keys,
    version,
)

# default KFP artifacts and output (ui metadata, metrics etc.)
# directories to /tmp to allow running with security context
KFPMETA_DIR = "/tmp"
KFP_ARTIFACTS_DIR = "/tmp"


class PipelineRunType:
    run = "run"
    build = "build"
    deploy = "deploy"


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
    returns: list[Union[str, dict[str, str]]] = None,
    auto_build: bool = False,
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
    :param outputs:  dictionary of output objects + optional paths (if path is
                     omitted the path will be the out_path/key.
    :param in_path:  default input path/url (prefix) for inputs
    :param out_path: default output path/url (prefix) for artifacts
    :param rundb:    path for rundb (or use 'MLRUN_DBPATH' env instead)
    :param mode:     run mode, e.g. 'pass' for using the command without mlrun wrapper
    :param handler   code entry-point/handler name
    :param job_image name of the image user for the job
    :param verbose:  add verbose prints/logs
    :param scrape_metrics:  whether to add the `mlrun/scrape-metrics` label to this run's resources
    :param returns: List of configurations for how to log the returning values from the handler's run (as artifacts or
                    results). The list's length must be equal to the amount of returning objects. A configuration may be
                    given as:

                    * A string of the key to use to log the returning value as result or as an artifact. To specify
                      The artifact type, it is possible to pass a string in the following structure:
                      "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no artifact
                      type is specified, the object's default artifact type will be used.
                    * A dictionary of configurations to use when logging. Further info per object type and artifact
                      type can be given there. The artifact key must appear in the dictionary as "key": "the_key".
    :param auto_build: when set to True and the function require build it will be built on the first
                       function run, use only if you dont plan on changing the build config between runs

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

        # feed 1st step results into the second step
        validate = mlrun_validate(
            train.outputs['model-txt']).apply(mount_v3io())

    """
    from mlrun_pipelines.ops import generate_pipeline_node

    secrets = [] if secrets is None else secrets
    params = {} if params is None else params
    hyperparams = {} if hyperparams is None else hyperparams
    if hyper_param_options and isinstance(hyper_param_options, dict):
        hyper_param_options = HyperParamOptions.from_dict(hyper_param_options)
    inputs = {} if inputs is None else inputs
    returns = [] if returns is None else returns
    outputs = [] if outputs is None else outputs
    labels = {} if labels is None else labels

    rundb = rundb or mlrun.db.get_or_set_dburl()
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
        returns = returns or runobj.spec.returns
        outputs = outputs or runobj.spec.outputs
        in_path = in_path or runobj.spec.input_path
        out_path = out_path or runobj.spec.output_path
        secrets = secrets or runobj.spec.secret_sources
        project = project or runobj.metadata.project
        labels = runobj.metadata.labels or labels
        verbose = verbose or runobj.spec.verbose
        scrape_metrics = scrape_metrics or runobj.spec.scrape_metrics

    outputs = RunSpec.join_outputs_and_returns(outputs=outputs, returns=returns)

    if not name:
        if not function_name:
            raise ValueError("name or function object must be specified")
        name = function_name
        if handler:
            short_name = handler
            for separator in ["#", "::", "."]:
                # drop paths, module or class name from short name
                if separator in short_name:
                    short_name = short_name.split(separator)[-1]
            name += "-" + short_name

    if hyperparams or param_file:
        outputs.append("iteration_results")
    if "run_id" not in outputs:
        outputs.append("run_id")

    params = params or {}
    hyperparams = hyperparams or {}
    inputs = inputs or {}
    returns = returns or []
    secrets = secrets or []

    mlrun.runtimes.utils.enrich_run_labels(labels)

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
    for log_hint in returns:
        cmd += [
            "--returns",
            json.dumps(log_hint) if isinstance(log_hint, dict) else log_hint,
        ]
    for label, val in labels.items():
        cmd += ["--label", f"{label}={val}"]
    for output in outputs:
        cmd += ["-o", str(output)]
        file_outputs[output.replace(".", "_")] = (
            f"/tmp/{output}"  # not using path.join to avoid windows "\"
        )
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
    if auto_build:
        cmd += ["--auto-build"]
    if more_args:
        cmd += more_args

    registry = get_default_reg()
    if image and image.startswith("."):
        if registry:
            image = f"{registry}/{image[1:]}"
        else:
            raise ValueError("local image registry env not found")

    image = mlrun.utils.enrich_image_url(
        image, mlrun.get_version(), str(version.Version().get_python_version())
    )

    return generate_pipeline_node(
        project,
        name,
        image,
        cmd + [command],
        file_outputs,
        function,
        func_url,
        scrape_metrics,
        code_env,
        registry,
    )


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
    from mlrun_pipelines.ops import generate_image_builder_pipeline_node

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

    return generate_image_builder_pipeline_node(name, function, func_url, cmd)


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
    from mlrun_pipelines.ops import generate_deployer_pipeline_node

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

    return generate_deployer_pipeline_node(
        name,
        function,
        func_url,
        cmd,
    )


def get_default_reg():
    if config.httpdb.builder.docker_registry:
        return config.httpdb.builder.docker_registry
    namespace_domain = os.environ.get("IGZ_NAMESPACE_DOMAIN", None)
    if namespace_domain is not None:
        return f"docker-registry.{namespace_domain}:80"
    return ""


def format_summary_from_kfp_run(
    kfp_run, project=None, run_db: "mlrun.db.RunDBInterface" = None
):
    from mlrun_pipelines.ops import generate_kfp_dag_and_resolve_project

    override_project = project if project and project != "*" else None
    dag, project, message = generate_kfp_dag_and_resolve_project(
        kfp_run, override_project
    )
    run_id = kfp_run.id
    logger.debug("Formatting summary from KFP run", run_id=run_id, project=project)

    # run db parameter allows us to use the same db session for the whole flow and avoid session isolation issues
    if not run_db:
        run_db = mlrun.db.get_run_db()

    # enrich DAG with mlrun run info
    runs = run_db.list_runs(project=project, labels=f"workflow={run_id}")

    for run in runs:
        step = get_in(
            run,
            [
                "metadata",
                "labels",
                mlrun.common.constants.MLRunInternalLabels.runner_pod,
            ],
        )
        if step and step in dag:
            dag[step]["run_uid"] = get_in(run, "metadata.uid")
            dag[step]["kind"] = get_in(run, "metadata.labels.kind")
            error = get_in(run, "status.error")
            if error:
                dag[step]["error"] = error

    short_run = {
        "graph": dag,
        "run": mlrun.utils.helpers.format_run(kfp_run),
    }
    short_run["run"]["project"] = project
    short_run["run"]["message"] = message
    logger.debug("Completed summary formatting", run_id=run_id, project=project)
    return short_run


def show_kfp_run(run, clear_output=False):
    phase_to_color = {
        mlrun_pipelines.common.models.RunStatuses.failed: "red",
        mlrun_pipelines.common.models.RunStatuses.succeeded: "green",
        mlrun_pipelines.common.models.RunStatuses.skipped: "white",
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
            logger.warning(f"failed to plot graph, {err_to_str(exc)}")


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
    with open(os.path.join(KFPMETA_DIR, "mlpipeline-metrics.json"), "w") as f:
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
            # NOTE: if key has "../x", it would fail on path traversal
            path = os.path.join(KFP_ARTIFACTS_DIR, key)
            if not mlrun.utils.helpers.is_safe_path(KFP_ARTIFACTS_DIR, path):
                logger.warning(
                    "Path traversal is not allowed ignoring", path=path, key=key
                )
                continue
            path = os.path.abspath(path)
            logger.info("Writing artifact output", path=path, val=val)
            with open(path, "w") as fp:
                fp.write(str(val))
        except Exception as exc:
            logger.warning("Failed writing to temp file. Ignoring", exc=err_to_str(exc))
            pass

    text = "# Run Report\n"
    if "iterations" in struct["status"]:
        del struct["status"]["iterations"]

    text += "## Metadata\n```yaml\n" + dict_to_yaml(struct) + "```\n"

    metadata = {"outputs": [{"type": "markdown", "storage": "inline", "source": text}]}
    with open(os.path.join(KFPMETA_DIR, "mlpipeline-ui-metadata.json"), "w") as f:
        json.dump(metadata, f)


def get_kfp_outputs(artifacts, labels, project):
    outputs = []
    out_dict = {}
    for output in artifacts:
        key = output.get("metadata")["key"]
        output_spec = output.get("spec", {})

        target = output_spec.get("target_path", "")
        target = output_spec.get("inline", target)

        out_dict[key] = get_artifact_target(output, project=project)

        if target.startswith("v3io:///"):
            target = target.replace("v3io:///", "http://v3io-webapi:8081/")

        user = labels.get(
            mlrun.common.constants.MLRunInternalLabels.v3io_user, ""
        ) or os.environ.get("V3IO_USERNAME", "")
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
