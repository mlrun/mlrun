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
from copy import deepcopy
from os import environ

import mlrun
from .db import get_or_set_dburl
from .utils import run_keys, dict_to_yaml, logger, gen_md_table, get_artifact_target
from .config import config

KFPMETA_DIR = environ.get("KFPMETA_OUT_DIR", "")
KFP_ARTIFACTS_DIR = environ.get("KFP_ARTIFACTS_DIR", "/tmp")


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

    results["run_id"] = results.get("run_id", "{}/{}".format(project, uid))
    for key in struct["spec"].get(run_keys.outputs, []):
        val = "None"
        if key in out_dict:
            val = out_dict[key]
        elif key in results:
            val = results[key]
        try:
            path = f"{KFP_ARTIFACTS_DIR}/{key}"
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

        user = labels.get("v3io_user", "") or environ.get("V3IO_USERNAME", "")
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
                text = "## Dataset: {}  \n\n".format(key) + tbl_md
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
    tuning_strategy=None,
    verbose=None,
    scrape_metrics=False,
):
    """mlrun KubeFlow pipelines operator, use to form pipeline steps

    when using kubeflow pipelines, each step is wrapped in an mlrun_op
    one step can pass state and data to the next step, see example below.

    :param name:    name used for the step
    :param project: optional, project name
    :param image:   optional, run container image (will be executing the step)
                    the container should host all requiered packages + code
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
    :param param_file:  a csv file with parameter combinations, first row hold
                        the parameter names, following rows hold param values
    :param selector: selection criteria for hyperparams e.g. "max.accuracy"
    :param tuning_strategy: selection strategy for hyperparams e.g. list, grid, random
    :param labels:   labels to tag the job/run with ({key:val, ..})
    :param inputs:   dictionary of input objects + optional paths (if path is
                     omitted the path will be the in_path/key.
    :param outputs:  dictionary of input objects + optional paths (if path is
                     omitted the path will be the out_path/key.
    :param in_path:  default input path/url (prefix) for inputs
    :param out_path: default output path/url (prefix) for artifacts
    :param rundb:    path for rundb (or use 'MLRUN_DBPATH' env instead)
    :param mode:     run mode, e.g. 'noctx' for pushing params as args
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
                    out_path ='v3io:///bigdata/mlrun/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    # use data from the first step
    def mlrun_validate(modelfile):
        return mlrun_op('validation',
                    command = '/User/kubeflow/validation.py',
                    inputs = {'model.txt':modelfile},
                    out_path ='v3io:///bigdata/mlrun/{{workflow.uid}}/',
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
    from kfp import dsl
    from os import environ
    from kubernetes import client as k8s_client

    secrets = [] if secrets is None else secrets
    params = {} if params is None else params
    hyperparams = {} if hyperparams is None else hyperparams
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
                code_env = "{}".format(function.spec.build.functionSourceCode)
            else:
                runtime = "{}".format(function.to_dict())

        function_name = function.metadata.name
        if function.kind == "dask":
            image = (
                image
                or function.spec.kfp_image
                or "mlrun/ml-base:{}".format(config.version)
            )

    image = image or config.kfp_image

    if runobj:
        handler = handler or runobj.spec.handler_name
        params = params or runobj.spec.parameters
        hyperparams = hyperparams or runobj.spec.hyperparams
        param_file = param_file or runobj.spec.param_file
        tuning_strategy = tuning_strategy or runobj.spec.tuning_strategy
        selector = selector or runobj.spec.selector
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
        labels["v3io_user"] = environ.get("V3IO_USERNAME")
    if "owner" not in labels:
        labels["owner"] = environ.get("V3IO_USERNAME") or getpass.getuser()

    if name:
        cmd += ["--name", name]
    if func_url:
        cmd += ["-f", func_url]
    for secret in secrets:
        cmd += ["-s", "{}={}".format(secret["kind"], secret["source"])]
    for param, val in params.items():
        cmd += ["-p", "{}={}".format(param, val)]
    for xpram, val in hyperparams.items():
        cmd += ["-x", "{}={}".format(xpram, val)]
    for input_param, val in inputs.items():
        cmd += ["-i", "{}={}".format(input_param, val)]
    for label, val in labels.items():
        cmd += ["--label", "{}={}".format(label, val)]
    for o in outputs:
        cmd += ["-o", "{}".format(o)]
        file_outputs[o.replace(".", "_")] = "/tmp/{}".format(o)
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
    if tuning_strategy:
        cmd += ["--tuning-strategy", tuning_strategy]
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
            image = "{}/{}".format(registry, image[1:])
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
    # if rundb:
    #     cop.container.add_env_variable(k8s_client.V1EnvVar(
    #         name='MLRUN_DBPATH', value=rundb))
    if code_env:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_EXEC_CODE", value=code_env)
        )
    if registry:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(name="DEFAULT_DOCKER_REGISTRY", value=registry)
        )
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

    if config.mpijob_crd_version:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="MLRUN_MPIJOB_CRD_VERSION", value=config.mpijob_crd_version
            )
        )

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
                cmd += ["-m", "{}={}".format(m["key"], m["model_path"])]

    if env:
        for key, val in env.items():
            cmd += ["--env", "{}={}".format(key, val)]

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

    from kfp import dsl
    from os import environ
    from kubernetes import client as k8s_client

    cmd = ["python", "-m", "mlrun", "build", "--kfp"]
    if function:
        if not hasattr(function, "to_dict"):
            raise ValueError("function must specify a function runtime object")
        cmd += ["-r", "{}".format(function.to_dict())]
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

    if "DEFAULT_DOCKER_REGISTRY" in environ:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="DEFAULT_DOCKER_REGISTRY",
                value=environ.get("DEFAULT_DOCKER_REGISTRY"),
            )
        )
    if "IGZ_NAMESPACE_DOMAIN" in environ:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="IGZ_NAMESPACE_DOMAIN", value=environ.get("IGZ_NAMESPACE_DOMAIN")
            )
        )

    is_v3io = function.spec.build.source and function.spec.build.source.startswith(
        "v3io"
    )
    if "V3IO_ACCESS_KEY" in environ and is_v3io:
        cop.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="V3IO_ACCESS_KEY", value=environ.get("V3IO_ACCESS_KEY")
            )
        )

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
    return cop


def get_default_reg():
    if "DEFAULT_DOCKER_REGISTRY" in environ:
        return environ.get("DEFAULT_DOCKER_REGISTRY")
    if "IGZ_NAMESPACE_DOMAIN" in environ:
        return "docker-registry.{}:80".format(environ.get("IGZ_NAMESPACE_DOMAIN"))
    return ""
