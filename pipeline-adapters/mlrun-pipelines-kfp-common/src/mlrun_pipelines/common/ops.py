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
import typing

if typing.TYPE_CHECKING:
    from mlrun.secrets import SecretsStore


import io
import json
import multiprocessing
import os
import warnings
import zipfile
from ast import literal_eval
from copy import deepcopy
from typing import Union

import mlrun_pipelines.common.models
import yaml
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector

import mlrun
import mlrun.common.constants
import mlrun.common.schemas
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.model import HyperParamOptions, RunSpec
from mlrun.utils import (
    create_ipython_display,
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
    :param rundb:    Deprecated. use 'MLRUN_DBPATH' env instead.
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
                    out_path ='v3io:///projects/my-proj/mlrun/{{workflow.uid}}/')

    # use data from the first step
    def mlrun_validate(modelfile):
        return mlrun_op('validation',
                    command = '/User/kubeflow/validation.py',
                    inputs = {'model.txt':modelfile},
                    out_path ='v3io:///projects/my-proj/{{workflow.uid}}/')

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

    if rundb:
        warnings.warn(
            "rundb parameter is deprecated and will be removed in 1.9.0. "
            "use 'MLRUN_DBPATH' env instead.",
            DeprecationWarning,
        )

    secrets = [] if secrets is None else secrets
    params = {} if params is None else params
    hyperparams = {} if hyperparams is None else hyperparams
    if hyper_param_options and isinstance(hyper_param_options, dict):
        hyper_param_options = HyperParamOptions.from_dict(hyper_param_options)
    inputs = {} if inputs is None else inputs
    returns = [] if returns is None else returns
    outputs = [] if outputs is None else outputs
    labels = {} if labels is None else labels

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


def format_summary_from_kfp_run(kfp_run, project=None):
    from mlrun_pipelines.ops import generate_kfp_dag_and_resolve_project

    override_project = project if project and project != "*" else None
    dag, project, message = generate_kfp_dag_and_resolve_project(
        kfp_run, override_project
    )
    run_id = kfp_run.id
    logger.debug("Formatting summary from KFP run", run_id=run_id, project=project)

    # enrich DAG with mlrun run info
    runs = mlrun.db.get_run_db().list_runs(project=project, labels=f"workflow={run_id}")
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


def show_kfp_run(run, html_display_id=None, dag_display_id=None, with_html=True):
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

            run_id = run["run"]["id"]
            url = get_workflow_url(run["run"]["project"], run_id)
            href = f'<a href="{url}" target="_blank"><b>click here</b></a>'
            html = IPython.display.HTML(
                f"<div>Pipeline running (id={run_id}), {href} to view the details in MLRun UI</div>"
            )

            # Use externally supplied displays if this method was run as part of an 'animation' loop.
            # Or create new displays if this method was run once as a standalone.
            if with_html:
                html_display_id = html_display_id or create_ipython_display()
                IPython.display.update_display(html, display_id=html_display_id)

            dag_display_id = dag_display_id or create_ipython_display()
            IPython.display.update_display(dag, display_id=dag_display_id)

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
    _sanitize_ui_metadata(struct)
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


def _sanitize_ui_metadata(struct):
    status_fields_to_remove = ["iterations", "artifacts"]
    for field in status_fields_to_remove:
        struct["status"].pop(field, None)


def _enrich_node_selector(function):
    function_node_selector = getattr(function.spec, "node_selector") or {}
    project_node_selector = {}
    project = mlrun.get_current_project()
    if not project or (function.metadata.project != project.metadata.name):
        project = function._get_db().get_project(function.metadata.project)
    if project:
        project_node_selector = project.spec.default_function_node_selector

    function_node_selector = mlrun.runtimes.utils.resolve_node_selectors(
        project_node_selector, function_node_selector
    )
    return mlrun.utils.helpers.to_non_empty_values_dict(function_node_selector)


def replace_kfp_plaintext_secret_env_vars_with_secret_refs(
    byte_buffer: bytes,
    content_type: str,
    env_var_names: list[str],
    secrets_store: "SecretsStore",
) -> bytes:
    if content_type.endswith(
        "zip"
    ):  # The kfp workflow can also be delivered as a zip package containing
        # the workflow pipeline yaml as well as script and resource files.
        modified_zip_bytes = _enrich_kfp_workflow_credentials_in_subprocess(
            byte_buffer=byte_buffer,
            env_var_names=env_var_names,
            secrets_store=secrets_store,
        )
        return modified_zip_bytes
    elif content_type.endswith(("yaml", "plain")):
        modified_yaml_bytes = _enrich_kfp_workflow_yaml_credentials(
            yaml_bytes=byte_buffer,
            env_var_names=env_var_names,
            secrets_store=secrets_store,
        )
        return modified_yaml_bytes
    else:
        raise ValueError(f"Unsupported content type {content_type}")


def _enrich_kfp_workflow_credentials_in_subprocess(
    byte_buffer: bytes,
    env_var_names: list[str],
    secrets_store: "SecretsStore",
) -> bytes:
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_enrich_wrapper,
        args=(queue, byte_buffer, env_var_names, secrets_store),
    )
    process.start()
    result = queue.get()
    process.join()
    return result


def _enrich_wrapper(
    queue: multiprocessing.Queue,
    byte_buffer: bytes,
    env_var_names: list[str],
    secrets_store: "SecretsStore",
):
    result = _enrich_kfp_workflow_zip_credentials(
        byte_buffer=byte_buffer,
        env_var_names=env_var_names,
        secrets_store=secrets_store,
    )
    queue.put(result)


def _enrich_kfp_workflow_zip_credentials(
    byte_buffer: bytes,
    env_var_names: list[str],
    secrets_store: "SecretsStore",
) -> bytes:
    in_memory_zip = io.BytesIO(byte_buffer)
    with zipfile.ZipFile(in_memory_zip, "r") as zip_read:
        file_list = zip_read.namelist()
        files_data = {}
        for file_name in file_list:
            with zip_read.open(file_name) as f:
                files_data[file_name] = f.read()

    for file_name, file_data in files_data.items():
        if file_name.endswith(("yaml", "plain")):
            modified_yaml = _enrich_kfp_workflow_yaml_credentials(
                yaml_bytes=file_data,
                env_var_names=env_var_names,
                secrets_store=secrets_store,
            )
            files_data[file_name] = modified_yaml

    out_memory_zip = io.BytesIO()
    with zipfile.ZipFile(out_memory_zip, "w") as zip_write:
        for file_name, file_data in files_data.items():
            zip_write.writestr(file_name, file_data)

    return out_memory_zip.getvalue()


def _enrich_kfp_workflow_yaml_credentials(
    yaml_bytes: bytes,
    env_var_names: list[str],
    secrets_store: "SecretsStore",
) -> bytes:
    """
    Modifies the given workflow YAML to add secret environment variables to container specifications.
    The function checks if the workflow uses Argo Workflows or Tekton Pipelines and injects the
    environment variables accordingly.
    """
    workflow_dict = yaml.safe_load(yaml_bytes)
    # Determine the KFP version by checking the 'apiVersion' field
    api_version = (
        workflow_dict.get("api_version") or workflow_dict.get("apiVersion", "").lower()
    )

    if "argoproj.io" in api_version:  # KFP Argo Workflow
        spec = workflow_dict.get("spec")
        if not spec:
            logger.warning("Missing spec, not modifying workflow")
            return yaml_bytes

        for template in spec.get("templates", []):
            container = template.get("container")
            if container is not None:
                _replace_secret_envs_in_argocd_template(
                    env_var_names=env_var_names,
                    container=container,
                    secrets_store=secrets_store,
                )

        return yaml.safe_dump(workflow_dict).encode()

    elif "tekton.dev" in api_version:  # KFP Tekton Pipeline
        for task in workflow_dict["spec"].get("tasks", []):
            if "name" in task:
                _replace_secret_envs_in_tekton_template(
                    env_var_names=env_var_names,
                    task=task,
                    secrets_store=secrets_store,
                )
        result = yaml.safe_dump(workflow_dict).encode()
        return result
    else:
        raise ValueError(
            f"Unknown or unsupported KFP version '{api_version}'. No changes made."
        )


def _replace_secret_envs_in_argocd_template(
    env_var_names: list[str],
    container: dict,
    secrets_store: "SecretsStore",
) -> None:
    """
    Replaces specified environment variables in the container with secret references.
    """
    secret_name_to_secret_ref = {}
    container_envs = container.get("env", [])
    container["env"] = _replace_env_vars_with_secrets(
        env_vars=container_envs,
        env_var_names=env_var_names,
        secret_name_to_secret_ref=secret_name_to_secret_ref,
        secrets_store=secrets_store,
    )

    cmd_parts = container.get("command", [])
    _replace_secret_vars_in_function_spec(
        cmd_parts=cmd_parts,
        env_var_names=env_var_names,
        secret_name_to_secret_ref=secret_name_to_secret_ref,
        secrets_store=secrets_store,
    )


def _replace_secret_envs_in_tekton_template(
    env_var_names: list[str],
    task: dict,
    secrets_store: "SecretsStore",
) -> None:
    secret_name_to_secret_ref = {}
    step_template = task.get("stepTemplate", {})
    step_template["env"] = _replace_env_vars_with_secrets(
        env_vars=step_template.get("env", []),
        env_var_names=env_var_names,
        secret_name_to_secret_ref=secret_name_to_secret_ref,
        secrets_store=secrets_store,
    )


def _replace_secret_vars_in_function_spec(
    cmd_parts: list[str],
    env_var_names: list[str],
    secret_name_to_secret_ref: dict[str, V1EnvVar],
    secrets_store: "SecretsStore",
) -> None:
    """
    Replaces specified environment variables in the function spec within cmd_parts.
    """
    for cmd_part_index, cmd_part in enumerate(cmd_parts):
        # When calling mlrun, a -r flag specifies the runtime spec.
        # Here we modify the runtime spec to use secret references instead of plain text values.
        if cmd_part == "-r" and cmd_part_index + 1 < len(cmd_parts):
            raw_func_data = cmd_parts[cmd_part_index + 1]
            try:
                func_data = literal_eval(raw_func_data)
            except (ValueError, SyntaxError):
                logger.warning("Invalid func data, skipping", func_data=raw_func_data)
                continue

            func_spec = func_data.get("spec", {})
            func_envs = func_spec.get("env", [])
            func_spec["env"] = _replace_env_vars_with_secrets(
                env_vars=func_envs,
                env_var_names=env_var_names,
                secret_name_to_secret_ref=secret_name_to_secret_ref,
                secrets_store=secrets_store,
            )
            cmd_parts[cmd_part_index + 1] = repr(func_data)
            break


def _create_secret_env_var_for_pipeline(
    name: str,
    value: str,
    secrets_store: "SecretsStore",
) -> V1EnvVar:
    secret_name = secrets_store.store_auth_secret(
        secret=mlrun.common.schemas.AuthSecretData(
            username=name,
            access_key=value,
        ),
    )
    env_var = V1EnvVar(
        name=name,
        value_from=V1EnvVarSource(
            secret_key_ref=V1SecretKeySelector(
                name=secret_name,
                key=mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                    "access_key"
                ),
            )
        ),
    )

    return env_var


def _replace_env_vars_with_secrets(
    env_vars: list[dict],
    env_var_names: list[str],
    secret_name_to_secret_ref: dict[str, V1EnvVar],
    secrets_store: "SecretsStore",
) -> list[dict]:
    """
    Helper function to replace environment variables with secrets.
    """
    for env_var_index, env_var in enumerate(env_vars):
        env_var_name = env_var.get("name")
        if env_var_name in env_var_names:
            if env_var_name in secret_name_to_secret_ref:
                secret_env_var = secret_name_to_secret_ref[env_var_name]
            else:
                value = env_var.get("value")
                if value is None:
                    logger.warning("Skipping empty secret value")
                    continue
                secret_env_var = _create_secret_env_var_for_pipeline(
                    name=env_var_name,
                    value=value,
                    secrets_store=secrets_store,
                )
                secret_name_to_secret_ref[env_var_name] = secret_env_var
            env_vars[env_var_index] = _create_env_for_container(secret_env_var)
    return env_vars


def _create_env_for_container(env_var: V1EnvVar) -> V1EnvVar:
    if env_var.value_from is not None and env_var.value_from.secret_key_ref is not None:
        env_var = {
            "name": env_var.name,
            "valueFrom": {
                "secretKeyRef": {
                    "name": env_var.value_from.secret_key_ref.name,
                    "key": env_var.value_from.secret_key_ref.key,
                }
            },
        }
    else:
        env_var = {
            "name": env_var.name,
            "value": env_var.value,
        }
    return env_var
