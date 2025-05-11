#!/usr/bin/env python

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
import functools
import importlib.metadata
import json
import pathlib
import socket
import traceback
from ast import literal_eval
from base64 import b64decode
from os import environ, path, remove
from pprint import pprint

import click
import dotenv
import pandas as pd
import semver
import yaml
from tabulate import tabulate

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.platforms
import mlrun.utils.helpers
from mlrun.common.helpers import parse_versioned_object_uri
from mlrun.runtimes.mounts import auto_mount as auto_mount_modifier

from .config import config as mlconf
from .db import get_run_db
from .errors import err_to_str
from .model import RunTemplate
from .projects import load_project
from .run import (
    get_object,
    import_function,
    import_function_to_dict,
    load_func_code,
    new_function,
)
from .runtimes import RemoteRuntime, RunError, RuntimeKinds, ServingRuntime
from .secrets import SecretsStore
from .utils import (
    RunKeys,
    dict_to_yaml,
    get_in,
    is_relative_path,
    list2dict,
    logger,
    update_in,
)
from .utils.version import Version

pd.set_option("mode.chained_assignment", None)


def validate_base_argument(ctx: click.Context, param: click.Parameter, value: str):
    # click 8.2 expects the context to be passed to make_metavar
    if semver.VersionInfo.parse(
        importlib.metadata.version("click")
    ) < semver.VersionInfo.parse("8.2.0"):
        metavar_func = functools.partial(param.make_metavar)
    else:
        metavar_func = functools.partial(param.make_metavar, ctx)
    if value and value.startswith("-"):
        raise click.BadParameter(
            f"{param.human_readable_name} ({value}) cannot start with '-', ensure the command options are typed "
            f"correctly. Preferably use '--' to separate options and arguments "
            f"e.g. 'mlrun run --option1 --option2 -- {metavar_func()} [--arg1|arg1] [--arg2|arg2]'",
            ctx=ctx,
            param=param,
        )
    return value


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("url", type=str, required=False, callback=validate_base_argument)
@click.option(
    "--param",
    "-p",
    default=[],
    multiple=True,
    help="parameter name and value tuples, e.g. -p x=37 -p y='text'",
)
@click.option("--inputs", "-i", multiple=True, help="input artifact")
@click.option("--outputs", "-o", multiple=True, help="output artifact/result for kfp")
@click.option("--in-path", help="default input path/url (prefix) for artifact")
@click.option("--out-path", help="default output path/url (prefix) for artifact")
@click.option(
    "--secrets",
    "-s",
    multiple=True,
    help="secrets file=<filename> or env=ENV_KEY1,..",
)
@click.option("--uid", help="unique run ID")
@click.option("--name", help="run name")
@click.option(
    "--workflow", help="sets the run labels to match the given workflow name/id"
)
@click.option("--project", help="project name/id")
@click.option("--db", default="", help="save run results to path or DB url")
@click.option(
    "--runtime", "-r", default="", help="function spec dict, for pipeline usage"
)
@click.option(
    "--kfp", is_flag=True, help="running inside Kubeflow Piplines, do not use"
)
@click.option(
    "--hyperparam",
    "-x",
    default=[],
    multiple=True,
    help="hyper parameters (will expand to multiple tasks) e.g. --hyperparam p2=[1,2,3]",
)
@click.option(
    "--param-file", default="", help="path to csv table of execution (hyper) params"
)
@click.option(
    "--selector",
    default="",
    help="how to select the best result from a list, e.g. max.accuracy",
)
@click.option(
    "--hyper-param-strategy",
    default="",
    help="hyperparam tuning strategy list | grid | random",
)
@click.option(
    "--hyper-param-options",
    default="",
    help="hyperparam options json string",
)
@click.option(
    "--func-url",
    "-f",
    default="",
    help="path/url of function yaml or function " "yaml or db://<project>/<name>[:tag]",
)
@click.option("--task", default="", help="path/url to task yaml")
@click.option(
    "--handler", default="", help="invoke function handler inside the code file"
)
@click.option("--mode", help="special run mode ('pass' for using the command as is)")
@click.option("--schedule", help="cron schedule")
@click.option("--from-env", is_flag=True, help="read the spec from the env var")
@click.option("--dump", is_flag=True, help="dump run results as YAML")
@click.option("--image", default="", help="container image (defaults to mlrun/mlrun)")
@click.option("--kind", default="", help="serverless runtime kind")
@click.option("--source", default="", help="source code archive/git")
@click.option("--local", is_flag=True, help="run the task locally (ignore runtime)")
@click.option(
    "--auto-mount", is_flag=True, help="add volume mount to job using auto mount option"
)
@click.option("--workdir", default="", help="run working directory")
@click.option("--origin-file", default="", help="for internal use")
@click.option("--label", multiple=True, help="run labels (key=val)")
@click.option("--watch", "-w", is_flag=True, help="watch/tail run log")
@click.option("--verbose", is_flag=True, help="verbose log")
@click.option(
    "--scrape-metrics",
    is_flag=True,
    help="whether to add the `mlrun/scrape-metrics` label to this run's resources",
)
@click.option(
    "--env-file", default="", help="path to .env file to load config/variables from"
)
@click.option(
    "--auto-build",
    is_flag=True,
    help="when set functions will be built prior to run if needed",
)
@click.argument("run_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--ensure-project",
    is_flag=True,
    help="ensure the project exists, if not, create project",
)
@click.option(
    "--returns",
    multiple=True,
    help="Logging configurations for the handler's returning values",
)
def run(
    url,
    param,
    inputs,
    outputs,
    in_path,
    out_path,
    secrets,
    uid,
    name,
    workflow,
    project,
    db,
    runtime,
    kfp,
    hyperparam,
    param_file,
    selector,
    hyper_param_strategy,
    hyper_param_options,
    func_url,
    task,
    handler,
    mode,
    schedule,
    from_env,
    dump,
    image,
    kind,
    source,
    local,
    auto_mount,
    workdir,
    origin_file,
    label,
    watch,
    verbose,
    scrape_metrics,
    env_file,
    auto_build,
    run_args,
    ensure_project,
    returns,
):
    """Execute a task and inject parameters."""

    if env_file:
        mlrun.set_env_from_file(env_file)

    out_path = out_path or environ.get("MLRUN_ARTIFACT_PATH")
    config = environ.get("MLRUN_EXEC_CONFIG")
    if from_env and config:
        config = json.loads(config)
        runobj = RunTemplate.from_dict(config)
    elif task:
        obj = get_object(task)
        task = yaml.load(obj, Loader=yaml.FullLoader)
        runobj = RunTemplate.from_dict(task)
    else:
        runobj = RunTemplate()

    set_item(runobj.metadata, uid, "uid")
    set_item(runobj.metadata, name, "name")
    set_item(runobj.metadata, project, "project")

    if label:
        label_dict = list2dict(label)
        for k, v in label_dict.items():
            runobj.metadata.labels[k] = v

    if workflow:
        runobj.metadata.labels[mlrun_constants.MLRunInternalLabels.workflow] = workflow
        runobj.metadata.labels[mlrun_constants.MLRunInternalLabels.runner_pod] = (
            socket.gethostname()
        )

    if db:
        mlconf.dbpath = db

    # remove potential quotes from command
    eval_url = py_eval(url)
    url = eval_url if isinstance(eval_url, str) else url
    url_file = url
    url_args = ""
    if url:
        split = url.split(maxsplit=1)
        url_file = split[0]
        if len(split) > 1:
            url_args = split[1]

    if ensure_project and project:
        mlrun.get_or_create_project(
            name=project,
            context="./",
        )
    if func_url or kind:
        if func_url:
            runtime = func_url_to_runtime(func_url, ensure_project)
            kind = get_in(runtime, "kind", kind or "job")
            if runtime is None:
                exit(1)
        else:
            kind = kind or "job"
            runtime = {"kind": kind, "spec": {"image": image or "mlrun/mlrun"}}

        if kind not in ["", "local", "dask"] and url:
            if url_file and path.isfile(url_file):
                with open(url_file) as fp:
                    body = fp.read()
                based = mlrun.utils.helpers.encode_user_code(body)
                logger.info(f"packing code at {url_file}")
                update_in(runtime, "spec.build.functionSourceCode", based)
                url = f"main{pathlib.Path(url_file).suffix} {url_args}"
                update_in(runtime, "spec.build.code_origin", url_file)
    elif runtime:
        runtime = py_eval(runtime)
        runtime = mlrun.utils.helpers.as_dict(runtime)
        if not isinstance(runtime, dict):
            print(f"Runtime parameter must be a dict, not {type(runtime)}")
            exit(1)
    else:
        runtime = {}

    code = environ.get("MLRUN_EXEC_CODE")
    if get_in(runtime, "kind", "") == "dask":
        code = get_in(runtime, "spec.build.functionSourceCode", code)
    if from_env and code:
        code = b64decode(code).decode("utf-8")
        origin_file = pathlib.Path(
            get_in(runtime, "spec.build.origin_filename", origin_file)
        )
        if kfp:
            print(f"Code:\n{code}\n")
        suffix = pathlib.Path(url_file).suffix if url else ".py"

        # * is a placeholder for the url file when we want to use url args and let mlrun resolve the url file
        if (
            suffix != ".py"
            and mode != "pass"
            and url_file != "{codefile}"
            and url_file != "*"
        ):
            print(
                f"command/url ({url}) must specify a .py file when not in 'pass' mode"
            )
            exit(1)
        if mode == "pass":
            if "{codefile}" in url:
                url_file = origin_file.name or "codefile"
                url = url.replace("{codefile}", url_file)
            elif suffix == ".sh" or origin_file.suffix == ".sh":
                url_file = origin_file.name or "codefile.sh"
                url = f"bash {url_file} {url_args}".strip()
            else:
                print(
                    "Error: command must be specified with '{codefile}' in it "
                    "(to determine the position of the code file)"
                )
                exit(1)
        else:
            url_file = "main.py"
            if origin_file.name:
                url_file = origin_file.stem + ".py"
            url = f"{url_file} {url_args}".strip()
        with open(url_file, "w") as fp:
            fp.write(code)

    # at this point the url placeholder should have been resolved to the actual url file
    if url == "*":
        print("command/url '*' placeholder is not allowed when code is not from env")
        exit(1)

    if url:
        if not name and not runtime:
            name = path.splitext(path.basename(url))[0]
            runobj.metadata.name = runobj.metadata.name or name
        update_in(runtime, "spec.command", url)

    if run_args:
        update_in(runtime, "spec.args", list(run_args))

    update_in(runtime, "spec.image", image or "mlrun/mlrun", replace=bool(image))

    set_item(runobj.spec, handler, "handler")
    set_item(runobj.spec, param, "parameters", fill_params(param))

    set_item(runobj.spec, hyperparam, "hyperparams", fill_params(hyperparam))
    if hyper_param_options:
        runobj.spec.hyper_param_options = py_eval(hyper_param_options)
    set_item(runobj.spec.hyper_param_options, param_file, "param_file")
    set_item(runobj.spec.hyper_param_options, hyper_param_strategy, "strategy")
    set_item(runobj.spec.hyper_param_options, selector, "selector")

    set_item(runobj.spec, inputs, RunKeys.inputs, list2dict(inputs))
    set_item(
        runobj.spec, returns, RunKeys.returns, [py_eval(value) for value in returns]
    )
    set_item(runobj.spec, in_path, RunKeys.input_path)
    set_item(runobj.spec, out_path, RunKeys.output_path)
    set_item(runobj.spec, outputs, RunKeys.outputs, list(outputs))
    set_item(
        runobj.spec, secrets, RunKeys.secrets, line2keylist(secrets, "kind", "source")
    )
    set_item(runobj.spec, verbose, "verbose")
    set_item(runobj.spec, scrape_metrics, "scrape_metrics")
    update_in(runtime, "metadata.name", name, replace=False)
    update_in(runtime, "metadata.project", project, replace=False)
    if not kind and "." in handler:
        # handle the case of module.submodule.handler
        update_in(runtime, "kind", "local")

    if kfp or runobj.spec.verbose or verbose:
        print(f"MLRun version: {str(Version().get())}")
        print("Runtime:")
        pprint(runtime)
        print("Run:")
        pprint(runobj.to_dict())

    try:
        fn = new_function(runtime=runtime, kfp=kfp, mode=mode, source=source)
        if workdir:
            fn.spec.workdir = workdir
        if auto_mount:
            fn.apply(auto_mount_modifier())
        fn.is_child = from_env and not kfp
        if kfp:
            # if pod is running inside kfp pod, we don't really need the run logs to be printed actively, we can just
            # pull the run state, and pull the logs periodically
            # we will set watch=None only when the pod is running inside kfp, and this tells the run to pull state
            # and logs periodically
            # TODO: change watch to be a flag with more options (with_logs, wait_for_completion, etc.)
            watch = watch or None
        resp = fn.run(
            runobj,
            watch=watch,
            schedule=schedule,
            local=local,
            auto_build=auto_build,
            project=project,
        )
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print(f"Runtime error: {err_to_str(err)}")
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("func_url", type=str, required=False, callback=validate_base_argument)
@click.option("--name", help="function name")
@click.option("--project", help="project name")
@click.option("--tag", default="", help="function tag")
@click.option("--image", "-i", help="target image path")
@click.option(
    "--source", "-s", default="", help="location/url of the source files dir/tar"
)
@click.option("--base-image", "-b", help="base docker image")
@click.option(
    "--command",
    "-c",
    default=[],
    multiple=True,
    help="build commands, e.g. '-c pip install pandas'",
)
@click.option("--secret-name", default="", help="container registry secret name")
@click.option("--archive", "-a", default="", help="destination archive for code (tar)")
@click.option("--silent", is_flag=True, help="do not show build logs")
@click.option("--with-mlrun", is_flag=True, help="add MLRun package")
@click.option("--db", default="", help="save run results to DB url")
@click.option(
    "--runtime", "-r", default="", help="function spec dict, for pipeline usage"
)
@click.option(
    "--kfp", is_flag=True, help="running inside Kubeflow Pipelines, do not use"
)
@click.option("--skip", is_flag=True, help="skip if already deployed")
@click.option(
    "--env-file", default="", help="path to .env file to load config/variables from"
)
@click.option(
    "--ensure-project",
    is_flag=True,
    help="ensure the project exists, if not, create project",
)
@click.option(
    "--state-file-path", default="/tmp/state", help="path to file with state data"
)
@click.option(
    "--image-file-path", default="/tmp/image", help="path to file with image data"
)
@click.option(
    "--full-image-file-path",
    default="/tmp/fullimage",
    help="path to file with full image data",
)
def build(
    func_url,
    name,
    project,
    tag,
    image,
    source,
    base_image,
    command,
    secret_name,
    archive,
    silent,
    with_mlrun,
    db,
    runtime,
    kfp,
    skip,
    env_file,
    ensure_project,
    state_file_path,
    image_file_path,
    full_image_file_path,
):
    """Build a container image from code and requirements."""

    if env_file:
        mlrun.set_env_from_file(env_file)

    if db:
        mlconf.dbpath = db

    if runtime:
        runtime = py_eval(runtime)
        runtime = mlrun.utils.helpers.as_dict(runtime)
        if not isinstance(runtime, dict):
            print(f"Runtime parameter must be a dict, not {type(runtime)}")
            exit(1)
        if kfp:
            print("Runtime:")
            pprint(runtime)
        # use kind = "job" by default if not specified
        runtime.setdefault("kind", "job")
        func = new_function(runtime=runtime)

    elif func_url:
        if func_url.startswith("db://"):
            func_url = func_url[5:]
        elif func_url == ".":
            func_url = "function.yaml"
        func = import_function(func_url)

    else:
        print("Error: Function path or url are required")
        exit(1)

    meta = func.metadata
    meta.project = project or meta.project or mlconf.default_project
    meta.name = name or meta.name
    meta.tag = tag or meta.tag

    b = func.spec.build
    if func.kind not in ["", "local"]:
        b.base_image = base_image or b.base_image
        b.commands = list(command) or b.commands
        b.image = image or b.image
        b.secret = secret_name or b.secret

    if source.endswith(".py"):
        if not path.isfile(source):
            print(f"Source file doesnt exist ({source})")
            exit(1)
        with open(source) as fp:
            body = fp.read()
        based = mlrun.utils.helpers.encode_user_code(body)
        logger.info(f"Packing code at {source}")
        b.functionSourceCode = based
        func.spec.command = ""
    else:
        b.source = source or b.source
        # todo: upload stuff

    archive = archive or mlconf.default_archive
    if archive:
        src = b.source or "./"
        logger.info(f"Uploading data from {src} to {archive}")
        target = archive if archive.endswith("/") else archive + "/"
        target += f"src-{meta.project}-{meta.name}-{meta.tag or 'latest'}.tar.gz"
        mlrun.datastore.utils.upload_tarball(src, target)
        # todo: replace function.yaml inside the tar
        b.source = target

    with_mlrun = True if with_mlrun else None  # False will map to None

    if ensure_project and project:
        mlrun.get_or_create_project(
            name=project,
            context="./",
        )

    if hasattr(func, "deploy"):
        logger.info("Remote deployment started")
        try:
            func.deploy(
                with_mlrun=with_mlrun, watch=not silent, is_kfp=kfp, skip_deployed=skip
            )
        except Exception as err:
            print(f"Deploy error, {err_to_str(err)}")
            exit(1)

        state = func.status.state
        image = func.spec.image
        if kfp:
            with open(state_file_path, "w") as fp:
                fp.write(state or "none")
            full_image = func.full_image_path(image) or ""
            with open(image_file_path, "w") as fp:
                fp.write(image)
            with open(full_image_file_path, "w") as fp:
                fp.write(full_image)
            print("Full image path = ", full_image)

        print(f"Function built, state={state} image={image}")
    else:
        print("Function does not have a deploy() method")
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("spec", type=str, required=False, callback=validate_base_argument)
@click.option("--source", "-s", default="", help="location/url of the source")
@click.option(
    "--func-url",
    "-f",
    default="",
    help="path/url of function yaml or function " "yaml or db://<project>/<name>[:tag]",
)
@click.option("--project", "-p", default="", help="project name")
@click.option("--model", "-m", multiple=True, help="model name and path (name=path)")
@click.option("--kind", "-k", default=None, help="runtime sub kind")
@click.option("--tag", default="", help="version tag")
@click.option("--env", "-e", multiple=True, help="environment variables")
@click.option("--verbose", is_flag=True, help="verbose log")
@click.option(
    "--env-file", default="", help="path to .env file to load config/variables from"
)
@click.option(
    "--ensure-project",
    is_flag=True,
    help="ensure the project exists, if not, create project",
)
def deploy(
    spec,
    source,
    func_url,
    project,
    model,
    tag,
    kind,
    env,
    verbose,
    env_file,
    ensure_project,
):
    """Deploy model or function"""
    if env_file:
        mlrun.set_env_from_file(env_file)

    if ensure_project and project:
        mlrun.get_or_create_project(
            name=project,
            context="./",
        )

    if func_url:
        runtime = func_url_to_runtime(func_url, ensure_project)
        if runtime is None:
            exit(1)
    elif spec:
        runtime = py_eval(spec)
    else:
        runtime = {}

    runtime = mlrun.utils.helpers.as_dict(runtime)
    if not isinstance(runtime, dict):
        print(f"Runtime parameter must be a dict, not {type(runtime)}")
        exit(1)

    if verbose:
        pprint(runtime)
        pprint(model)

    # support both v1 & v2+ model struct for backwards compatibility
    if runtime and runtime["kind"] == RuntimeKinds.serving:
        print("Deploying V2 model server")
        function = ServingRuntime.from_dict(runtime)
        if model:
            # v2+ model struct (list of json obj)
            for _model in model:
                args = json.loads(_model)
                function.add_model(**args)
    else:
        function = RemoteRuntime.from_dict(runtime)
        if kind:
            function.spec.function_kind = kind
        if model:
            # v1 model struct (list of k=v)
            models = list2dict(model)
            for k, v in models.items():
                function.add_model(k, v)

    function.spec.source = source
    if env:
        for k, v in list2dict(env).items():
            function.set_env(k, v)
    function.verbose = verbose

    try:
        addr = function.deploy(project=project, tag=tag)
    except Exception as err:
        print(f"deploy error: {err_to_str(err)}")
        exit(1)

    print(f"Function deployed, address={addr}")
    with open("/tmp/output", "w") as fp:
        fp.write(addr)
    with open("/tmp/name", "w") as fp:
        fp.write(function.status.nuclio_name)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("kind", type=str, callback=validate_base_argument)
@click.argument(
    "name",
    type=str,
    default="",
    required=False,
    callback=validate_base_argument,
)
@click.option("--selector", "-s", default="", help="label selector")
@click.option("--namespace", "-n", help="kubernetes namespace")
@click.option("--uid", help="unique ID")
@click.option("--project", "-p", help="project name")
@click.option("--tag", "-t", default="", help="artifact/function tag")
@click.option("--db", help="db path/url")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def get(kind, name, selector, namespace, uid, project, tag, db, extra_args):
    """List/get one or more object per kind/class.

    KIND - resource type to list/get: run | runtime | workflow | artifact | function
    NAME - optional, resource name or category
    """

    if db:
        mlconf.dbpath = db
    if not project:
        logger.warning(
            "Project parameter was not specified. Defaulting to 'default' project"
        )
    if kind.startswith("po"):
        logger.warning("Unsupported, use 'get runtimes' instead")
        return

    elif kind.startswith("runtime"):
        run_db = get_run_db(db or mlconf.dbpath)
        # the name field is used as function kind, set to None if empty
        name = name if name else None
        runtimes = run_db.list_runtime_resources(
            label_selector=selector, kind=name, project=project
        )
        for runtime in runtimes:
            print(dict_to_yaml(runtime.dict()))
            print()

    elif kind.startswith("run"):
        if tag:
            print(
                "Unsupported argument '--tag' for listing runs. Perhaps you should use '--selector' instead"
            )
            return

        run_db = get_run_db()
        if name:
            run = run_db.read_run(name, project=project)
            print(dict_to_yaml(run))
            return

        runs = run_db.list_runs(uid=uid, project=project, labels=selector)
        df = runs.to_df()[
            ["name", "uid", "iter", "start", "end", "state", "parameters", "results"]
        ]
        # df['uid'] = df['uid'].apply(lambda x: f'..{x[-6:]}')
        for time_column in ["start", "end"]:
            df[time_column] = df[time_column].apply(time_str)
        df["parameters"] = df["parameters"].apply(dict_to_str)
        df["results"] = df["results"].apply(dict_to_str)
        print(tabulate(df, headers="keys"))

    elif kind.startswith("art"):
        run_db = get_run_db()
        artifacts = run_db.list_artifacts(
            name, project=project, tag=tag, labels=selector
        )
        df = artifacts.to_df()[
            ["key", "iter", "kind", "path", "hash", "updated", "uri", "tree"]
        ]
        df["tree"] = df["tree"].apply(lambda x: f"..{x[-8:]}")
        df["hash"] = df["hash"].apply(lambda x: f"..{x[-6:]}")
        df["updated"] = df["updated"].apply(time_str)
        df.rename(columns={"tree": "job/workflow uid"}, inplace=True)
        print(tabulate(df, headers="keys"))

    elif kind.startswith("func"):
        run_db = get_run_db()
        if name:
            f = run_db.get_function(name, project=project, tag=tag)
            print(dict_to_yaml(f))
            return

        functions = run_db.list_functions(name, project=project, labels=selector)
        lines = []
        headers = ["kind", "state", "name:tag", "hash"]
        for f in functions:
            name = get_in(f, "metadata.name")
            tag = get_in(f, "metadata.tag", "")
            line = [
                get_in(f, "kind", ""),
                get_in(f, "status.state", ""),
                f"{name}:{tag}",
                get_in(f, "metadata.hash", ""),
            ]
            lines.append(line)
        print(tabulate(lines, headers=headers))

    elif kind.startswith("workflow"):
        run_db = get_run_db()
        if project == "*":
            print("Warning, reading workflows for all projects may take a long time !")
            pipelines = run_db.list_pipelines(project=project, page_size=200)
            pipe_runs = pipelines.runs
            while pipelines.next_page_token is not None:
                pipelines = run_db.list_pipelines(
                    project=project, page_token=pipelines.next_page_token
                )
                pipe_runs.extend(pipelines.runs)
        else:
            pipelines = run_db.list_pipelines(project=project)
            pipe_runs = pipelines.runs

        lines = []
        headers = ["project", "name", "status", "created at", "finished at"]
        for pipe_run in pipe_runs:
            line = [
                pipe_run["project"],
                pipe_run["name"],
                pipe_run["status"],
                pipe_run["created_at"],
                pipe_run["finished_at"],
            ]
            lines.append(line)
        print(tabulate(lines, headers=headers))

    else:
        print(
            "Currently only get runs | runtimes | workflows | artifacts  | func [name] | runtime are supported"
        )


@main.command()
def version():
    """get mlrun version"""
    print(f"MLRun version: {str(Version().get())}")


@main.command()
@click.argument("uid", type=str)
@click.option(
    "--project", "-p", help="project name (defaults to mlrun.mlconf.default_project)"
)
@click.option("--offset", type=int, default=0, help="byte offset")
@click.option("--db", help="api and db service path/url")
def logs(uid, project, offset, db):
    """Get or watch task logs"""
    mldb = get_run_db(db or mlconf.dbpath)
    if mldb.kind == "http":
        state, _ = mldb.watch_log(uid, project, watch=False, offset=offset)
    else:
        state, text = mldb.get_log(uid, project, offset=offset)
        if text:
            print(text.decode())

    if state:
        print(f"Final state: {state}")


@main.command()
@click.argument("context", default="", type=str, required=False)
@click.option("--name", "-n", help="project name")
@click.option("--url", "-u", help="remote git or archive url")
@click.option("--run", "-r", help="run workflow name of .py file")
@click.option(
    "--arguments",
    "-a",
    default=[],
    multiple=True,
    help="Kubeflow pipeline arguments name and value tuples (with -r flag), e.g. -a x=6",
)
@click.option("--artifact-path", "-p", help="output artifacts path")
@click.option(
    "--param",
    "-x",
    default=[],
    multiple=True,
    help="mlrun project parameter name and value tuples, e.g. -p x=37 -p y='text'",
)
@click.option(
    "--secrets",
    "-s",
    multiple=True,
    help="secrets file=<filename> or env=ENV_KEY1,..",
)
@click.option("--namespace", help="k8s namespace")
@click.option("--db", help="api and db service path/url")
@click.option("--init-git", is_flag=True, help="for new projects init git context")
@click.option(
    "--clone", "-c", is_flag=True, help="force override/clone into the context dir"
)
@click.option("--sync", is_flag=True, help="sync functions into db")
@click.option(
    "--watch", "-w", is_flag=True, help="wait for pipeline completion (with -r flag)"
)
@click.option(
    "--dirty", "-d", is_flag=True, help="allow run with uncommitted git changes"
)
@click.option("--git-repo", help="git repo (org/repo) for git comments")
@click.option(
    "--git-issue", type=int, default=None, help="git issue number for git comments"
)
@click.option("--handler", default=None, help="workflow function handler name")
@click.option("--engine", default=None, help="workflow engine (kfp/local)")
@click.option("--local", is_flag=True, help="try to run workflow functions locally")
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="timeout in seconds to wait for pipeline completion (used when watch=True)",
)
@click.option(
    "--env-file", default="", help="path to .env file to load config/variables from"
)
@click.option(
    "--save/--no-save",
    default=True,
    help="create and save the project if not exist",
)
@click.option(
    "--schedule",
    type=str,
    default=None,
    help="To create a schedule define a standard crontab expression string."
    "for help see: "
    "https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron."
    "For using the pre-defined workflow's schedule, set --schedule 'true'",
)
@click.option(
    "--save-secrets",
    is_flag=True,
    help="Store the project secrets as k8s secrets",
)
@click.option(
    "--notifications",
    "--notification",
    "-nt",
    multiple=True,
    help="To have a notification for the run set notification file "
    "destination define: file=notification.json or a "
    'dictionary configuration e.g \'{"slack":{"webhook":"<webhook>"}}\'',
)
def project(
    context,
    name,
    url,
    run,
    arguments,
    artifact_path,
    param,
    secrets,
    namespace,
    db,
    init_git,
    clone,
    sync,
    watch,
    dirty,
    git_repo,
    git_issue,
    handler,
    engine,
    local,
    env_file,
    timeout,
    schedule,
    notifications,
    save_secrets,
    save,
):
    """load and/or run a project"""
    if env_file:
        mlrun.set_env_from_file(env_file)

    if db:
        mlconf.dbpath = db

    # set the CLI/GIT parameters in load_project() so they can be used by project setup scripts
    parameters = fill_params(param) if param else {}
    if git_repo:
        parameters["git_repo"] = git_repo
    if git_issue:
        parameters["git_issue"] = git_issue
    commit = environ.get("GITHUB_SHA") or environ.get("CI_COMMIT_SHA")
    if commit and not parameters.get("commit_id"):
        parameters["commit_id"] = commit

    proj = load_project(
        context,
        url,
        name,
        init_git=init_git,
        clone=clone,
        save=save,
        parameters=parameters,
    )
    url_str = " from " + url if url else ""
    print(f"Loading project {proj.name}{url_str} into {context}:\n")

    if is_relative_path(artifact_path):
        artifact_path = path.abspath(artifact_path)
    if secrets:
        secrets = line2keylist(secrets, "kind", "source")
        secret_store = SecretsStore.from_list(secrets)
        # Used to run a workflow with secrets in runtime, without using or storing k8s secrets.
        # To run a scheduled workflow or to use those secrets in other runs, save
        # the secrets in k8s and use the --save-secret flag
        proj._secrets = secret_store
        if save_secrets:
            proj.set_secrets(secret_store._secrets)
    print(proj.to_yaml())

    if run:
        if schedule is not None and schedule.lower() in ["1", "yes", "y", "t", "true"]:
            schedule = True
        workflow_path = None
        if run.endswith(".py"):
            workflow_path = run
            run = None

        args = None
        if arguments:
            args = fill_params(arguments)

        print(f"Running workflow {run} file: {workflow_path}")
        gitops = (
            git_issue
            or environ.get("GITHUB_EVENT_PATH")
            or environ.get("CI_MERGE_REQUEST_IID")
        )
        if gitops:
            proj.notifiers.add_notification(
                "git",
                {
                    "repo": git_repo,
                    "issue": git_issue,
                    "token": proj.get_param("GIT_TOKEN"),
                },
            )
        if notifications:
            load_notification(notifications, proj)
        try:
            proj.run(
                name=run,
                workflow_path=workflow_path,
                arguments=args,
                artifact_path=artifact_path,
                namespace=namespace,
                sync=sync,
                watch=watch,
                dirty=dirty,
                workflow_handler=handler,
                engine=engine,
                local=local,
                schedule=schedule,
                timeout=timeout,
            )
        except Exception as err:
            print(traceback.format_exc())
            send_workflow_error_notification(run, proj, err)
            exit(1)

    elif sync:
        print("Saving project functions to db ..")
        proj.sync_functions(save=True)


def validate_runtime_kind(ctx, param, value):
    possible_kinds = RuntimeKinds.runtime_with_handlers()
    if value is not None and value not in possible_kinds:
        raise click.BadParameter(
            f"kind must be one of {possible_kinds}", ctx=ctx, param=param
        )
    return value


@main.command()
@click.argument("kind", callback=validate_runtime_kind, default=None, required=False)
@click.argument("object_id", metavar="id", type=str, default=None, required=False)
@click.option("--api", help="api service url")
@click.option("--label-selector", "-ls", default="", help="label selector")
@click.option(
    "--force", "-f", is_flag=True, help="clean resources in non-terminal states as well"
)
@click.option(
    "--grace-period",
    "-gp",
    type=int,
    # When someone triggers the cleanup manually we assume they want runtime resources in terminal state to be removed
    # now, therefore not using here mlconf.runtime_resources_deletion_grace_period
    default=0,
    help="the grace period (in seconds) that will be given to runtime resources (after they're in terminal state) "
    "before cleaning them. Ignored when --force is given",
    show_default=True,
)
def clean(kind, object_id, api, label_selector, force, grace_period):
    """
    Clean jobs resources

    \b
    Examples:

        \b
        # Clean resources for all runs of all runtimes
        mlrun clean

        \b
        # Clean resources for all runs of a specific kind (e.g. job)
        mlrun clean job

        \b
        # Clean resources for specific job (by uid)
        mlrun clean mpijob 15d04c19c2194c0a8efb26ea3017254b
    """
    mldb = get_run_db(api or mlconf.dbpath)
    mldb.delete_runtime_resources(
        kind=kind,
        object_id=object_id,
        label_selector=label_selector,
        force=force,
        grace_period=grace_period,
    )


@main.command(name="watch-stream")
@click.argument("url", type=str)
@click.option(
    "--shard-ids",
    "-s",
    multiple=True,
    type=int,
    help="shard id to listen on (can be multiple)",
)
@click.option("--seek", help="where to start/seek (EARLIEST or LATEST)")
@click.option(
    "--interval",
    "-i",
    default=3,
    show_default=True,
    help="interval in seconds",
    type=int,
)
@click.option(
    "--is-json",
    "-j",
    is_flag=True,
    help="indicate the payload is json (will be deserialized)",
)
def watch_stream(url, shard_ids, seek, interval, is_json):
    """watch on a stream and print data every interval"""
    mlrun.platforms.watch_stream(
        url, shard_ids, seek, interval=interval, is_json=is_json
    )


@main.command(name="config")
@click.argument("command", type=str, default="", required=False)
@click.option(
    "--env-file",
    "-f",
    default="",
    help="path to the mlrun .env file (defaults to '~/.mlrun.env')",
)
@click.option("--api", "-a", type=str, help="api service url")
@click.option("--artifact-path", "-p", help="default artifacts path")
@click.option("--username", "-u", help="username (for remote access)")
@click.option("--access-key", "-k", help="access key (for remote access)")
@click.option(
    "--env-vars",
    "-e",
    default=[],
    multiple=True,
    help="additional env vars, e.g. -e AWS_ACCESS_KEY_ID=<key-id>",
)
def show_or_set_config(
    command, env_file, api, artifact_path, username, access_key, env_vars
):
    """get or set mlrun config

    \b
    Commands:
        get (default) - list the local or remote configuration
                        (can specify the remote api + credentials or an env_file)
        set           - set configuration parameters in mlrun default or specified .env file
        clear         - delete the default or specified config .env file

    \b
    Examples:
        # read the default config
        mlrun config
        # read config using an env file (with connection details)
        mlrun config -f mymlrun.env
        # set configuration and write it to the default env file (~/.mlrun.env)
        mlrun config set -a http://localhost:8080 -u joe -k mykey -e AWS_ACCESS_KEY_ID=<key-id>

    """
    op = command.lower()
    if not op or op == "get":
        # print out the configuration (default or based on the specified env/api)
        if env_file and not path.isfile(path.expanduser(env_file)):
            print(f"Error: Env file {env_file} does not exist")
            exit(1)
        if env_file or api:
            mlrun.set_environment(
                api,
                artifact_path=artifact_path,
                access_key=access_key,
                username=username,
                env_file=env_file,
            )
        print(mlconf.dump_yaml())

    elif op == "set":
        # update the env settings in the default or specified .env file
        filename = path.expanduser(env_file or mlrun.config.default_env_file)
        if not path.isfile(filename):
            print(
                f".env file {filename} not found, creating new and setting configuration"
            )
        else:
            print(f"Updating configuration in .env file {filename}")
        env_dict = {
            "MLRUN_DBPATH": api,
            "MLRUN_ARTIFACT_PATH": artifact_path,
            "V3IO_USERNAME": username,
            "V3IO_ACCESS_KEY": access_key,
        }
        for key, value in env_dict.items():
            if value:
                dotenv.set_key(filename, key, value, quote_mode="always")
        if env_vars:
            for key, value in list2dict(env_vars).items():
                dotenv.set_key(filename, key, value, quote_mode="always")
        if env_file:
            # if its not the default file print the usage details
            print(
                f"To use the {env_file} .env file add the following to your development environment:\n"
                f"MLRUN_ENV_FILE={env_file}"
            )

    elif op == "clear":
        filename = path.expanduser(env_file or mlrun.config.default_env_file)
        if not path.isfile(filename):
            print(f".env file {filename} not found")
        else:
            print(f"Deleting .env file {filename}")
            remove(filename)

    else:
        print(f"Error: Unsupported config option {op}")


def fill_params(params, params_dict=None):
    params_dict = params_dict or {}
    for param in params:
        i = param.find("=")
        if i == -1:
            continue
        key, value = param[:i].strip(), param[i + 1 :].strip()
        if key is None:
            raise ValueError(f"cannot find param key in line ({param})")
        params_dict[key] = py_eval(value)
    return params_dict


def py_eval(data):
    try:
        value = literal_eval(data)
        return value
    except (SyntaxError, ValueError):
        return data


def set_item(obj, item, key, value=None):
    if item:
        if value:
            setattr(obj, key, value)
        else:
            setattr(obj, key, item)


def line2keylist(lines: list, keyname="key", valname="path"):
    out = []
    for line in lines:
        i = line.find("=")
        if i == -1:
            raise ValueError(f'cannot find "=" in line ({keyname}={valname})')
        key, value = line[:i].strip(), line[i + 1 :].strip()
        if key is None:
            raise ValueError(f"cannot find key in line ({keyname}={valname})")
        value = path.expandvars(value)
        out += [{keyname: key, valname: value}]
    return out


def time_str(x):
    try:
        return x.strftime("%b %d %H:%M:%S")
    except ValueError:
        return ""


def dict_to_str(struct: dict):
    if not struct:
        return []
    return ",".join([f"{k}={v}" for k, v in struct.items()])


def func_url_to_runtime(func_url, ensure_project: bool = False):
    try:
        if func_url.startswith("db://"):
            func_url = func_url[5:]
            project_instance, name, tag, hash_key = parse_versioned_object_uri(func_url)
            run_db = get_run_db(mlconf.dbpath)
            runtime = run_db.get_function(name, project_instance, tag, hash_key)
        elif func_url == "." or func_url.endswith(".yaml"):
            func_url = "function.yaml" if func_url == "." else func_url
            runtime = import_function_to_dict(func_url, {})
        else:
            mlrun_project = load_project(".", save=ensure_project)
            function = mlrun_project.get_function(func_url, enrich=True)
            if function.kind == "local":
                command, function = load_func_code(function)
                function.spec.command = command
            runtime = function.to_dict()
    except Exception as exc:
        logger.error(f"function {func_url} not found, {err_to_str(exc)}")
        return None

    if not runtime:
        logger.error(f"function {func_url} not found or is null")
        return None

    return runtime


def load_notification(notifications: str, project: mlrun.projects.MlrunProject):
    """
    A dictionary or json file containing notification dictionaries can be used by the user to set notifications.
    Each notification is stored in a tuple called notifications.
    The code then goes through each value in the notifications tuple and check
    if the notification starts with "file=", such as "file=notification.json," in those cases it loads the
    notification.json file and uses add_notification_to_project to add the notifications from the file to
    the project. If not, it adds the notification dictionary to the project.
    :param notifications:  Notifications file or a dictionary to be added to the project
    :param project: The object to which the notifications will be added
    :return:
    """
    for notification in notifications:
        if notification.startswith("file="):
            file_path = notification.split("=")[-1]
            notification = open(file_path)
            notification = json.load(notification)
        else:
            notification = json.loads(notification)
        add_notification_to_project(notification, project)


def add_notification_to_project(
    notification: str, project: mlrun.projects.MlrunProject
):
    for notification_type, notification_params in notification.items():
        project.notifiers.add_notification(
            notification_type=notification_type, params=notification_params
        )


def send_workflow_error_notification(
    run_id: str, mlproject: mlrun.projects.MlrunProject, error: Exception
):
    message = (
        f":x: Failed to run scheduled workflow {run_id} in Project {mlproject.name} !\n"
        f"error: ```{err_to_str(error)}```"
    )
    mlproject.notifiers.push(
        message=message, severity=mlrun.common.schemas.NotificationSeverity.ERROR
    )


if __name__ == "__main__":
    main()
