#!/usr/bin/env python

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
import json
import pathlib
import socket
import traceback
from ast import literal_eval
from base64 import b64decode, b64encode
from os import environ, path, remove
from pprint import pprint
from subprocess import Popen
from sys import executable
from urllib.parse import urlparse

import click
import dotenv
import pandas as pd
import yaml
from tabulate import tabulate

import mlrun

from .builder import upload_tarball
from .config import config as mlconf
from .db import get_run_db
from .errors import err_to_str
from .k8s_utils import K8sHelper
from .model import RunTemplate
from .platforms import auto_mount as auto_mount_modifier
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
    dict_to_yaml,
    get_in,
    is_relative_path,
    list2dict,
    logger,
    parse_versioned_object_uri,
    run_keys,
    update_in,
)
from .utils.version import Version

pd.set_option("mode.chained_assignment", None)


def validate_base_argument(ctx, param, value):
    if value and value.startswith("-"):
        raise click.BadParameter(
            f"{param.human_readable_name} ({value}) cannot start with '-', ensure the command options are typed "
            f"correctly. Preferably use '--' to separate options and arguments "
            f"e.g. 'mlrun run --option1 --option2 -- {param.make_metavar()} [--arg1|arg1] [--arg2|arg2]'",
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
@click.option("--workflow", help="workflow name/id")
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
@click.option("--image", default="", help="container image")
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
        runobj.metadata.labels["workflow"] = workflow
        runobj.metadata.labels["mlrun/runner-pod"] = socket.gethostname()

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
    if func_url or kind or image:
        if func_url:
            runtime = func_url_to_runtime(func_url, ensure_project)
            kind = get_in(runtime, "kind", kind or "job")
            if runtime is None:
                exit(1)
        else:
            kind = kind or "job"
            runtime = {"kind": kind, "spec": {"image": image}}

        if kind not in ["", "local", "dask"] and url:
            if url_file and path.isfile(url_file):
                with open(url_file) as fp:
                    body = fp.read()
                based = b64encode(body.encode("utf-8")).decode("utf-8")
                logger.info(f"packing code at {url_file}")
                update_in(runtime, "spec.build.functionSourceCode", based)
                url = f"main{pathlib.Path(url_file).suffix} {url_args}"
                update_in(runtime, "spec.build.code_origin", url_file)
    elif runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print(f"runtime parameter must be a dict, not {type(runtime)}")
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
            print(f"code:\n{code}\n")
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
                    "error, command must be specified with '{codefile}' in it "
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
    if image:
        update_in(runtime, "spec.image", image)
    set_item(runobj.spec, handler, "handler")
    set_item(runobj.spec, param, "parameters", fill_params(param))

    set_item(runobj.spec, hyperparam, "hyperparams", fill_params(hyperparam))
    if hyper_param_options:
        runobj.spec.hyper_param_options = py_eval(hyper_param_options)
    set_item(runobj.spec.hyper_param_options, param_file, "param_file")
    set_item(runobj.spec.hyper_param_options, hyper_param_strategy, "strategy")
    set_item(runobj.spec.hyper_param_options, selector, "selector")

    set_item(runobj.spec, inputs, run_keys.inputs, list2dict(inputs))
    set_item(runobj.spec, in_path, run_keys.input_path)
    set_item(runobj.spec, out_path, run_keys.output_path)
    set_item(runobj.spec, outputs, run_keys.outputs, list(outputs))
    set_item(
        runobj.spec, secrets, run_keys.secrets, line2keylist(secrets, "kind", "source")
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
            runobj, watch=watch, schedule=schedule, local=local, auto_build=auto_build
        )
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print(f"runtime error: {err_to_str(err)}")
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
@click.option("--db", default="", help="save run results to path or DB url")
@click.option(
    "--runtime", "-r", default="", help="function spec dict, for pipeline usage"
)
@click.option(
    "--kfp", is_flag=True, help="running inside Kubeflow Piplines, do not use"
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
):
    """Build a container image from code and requirements."""

    if env_file:
        mlrun.set_env_from_file(env_file)

    if db:
        mlconf.dbpath = db

    if runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print(f"runtime parameter must be a dict, not {type(runtime)}")
            exit(1)
        if kfp:
            print("Runtime:")
            pprint(runtime)
        func = new_function(runtime=runtime)

    elif func_url:
        if func_url.startswith("db://"):
            func_url = func_url[5:]
        elif func_url == ".":
            func_url = "function.yaml"
        func = import_function(func_url)

    else:
        print("please specify the function path or url")
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
            print(f"source file doesnt exist ({source})")
            exit(1)
        with open(source) as fp:
            body = fp.read()
        based = b64encode(body.encode("utf-8")).decode("utf-8")
        logger.info(f"packing code at {source}")
        b.functionSourceCode = based
        func.spec.command = ""
    else:
        b.source = source or b.source
        # todo: upload stuff

    archive = archive or mlconf.default_archive
    if archive:
        src = b.source or "./"
        logger.info(f"uploading data from {src} to {archive}")
        target = archive if archive.endswith("/") else archive + "/"
        target += f"src-{meta.project}-{meta.name}-{meta.tag or 'latest'}.tar.gz"
        upload_tarball(src, target)
        # todo: replace function.yaml inside the tar
        b.source = target

    with_mlrun = True if with_mlrun else None  # False will map to None

    if ensure_project and project:
        mlrun.get_or_create_project(
            name=project,
            context="./",
        )

    if hasattr(func, "deploy"):
        logger.info("remote deployment started")
        try:
            func.deploy(
                with_mlrun=with_mlrun, watch=not silent, is_kfp=kfp, skip_deployed=skip
            )
        except Exception as err:
            print(f"deploy error, {err_to_str(err)}")
            exit(1)

        state = func.status.state
        image = func.spec.image
        if kfp:
            with open("/tmp/state", "w") as fp:
                fp.write(state or "none")
            full_image = func.full_image_path(image) or ""
            with open("/tmp/image", "w") as fp:
                fp.write(image)
            with open("/tmp/fullimage", "w") as fp:
                fp.write(full_image)
            print("full image path = ", full_image)

        print(f"function built, state={state} image={image}")
    else:
        print("function does not have a deploy() method")
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
@click.option("--dashboard", "-d", default="", help="nuclio dashboard url")
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
    dashboard,
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
    if not isinstance(runtime, dict):
        print(f"runtime parameter must be a dict, not {type(runtime)}")
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
        addr = function.deploy(dashboard=dashboard, project=project, tag=tag)
    except Exception as err:
        print(f"deploy error: {err_to_str(err)}")
        exit(1)

    print(f"function deployed, address={addr}")
    with open("/tmp/output", "w") as fp:
        fp.write(addr)
    with open("/tmp/name", "w") as fp:
        fp.write(function.status.nuclio_name)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pod", type=str, callback=validate_base_argument)
@click.option("--namespace", "-n", help="kubernetes namespace")
@click.option(
    "--timeout", "-t", default=600, show_default=True, help="timeout in seconds"
)
def watch(pod, namespace, timeout):
    """Read current or previous task (pod) logs."""
    print("This command will be deprecated in future version !!!\n")
    k8s = K8sHelper(namespace)
    status = k8s.watch(pod, namespace, timeout)
    print(f"Pod {pod} last status is: {status}")


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
        print("warning, project parameter was not specified using default !")
    if kind.startswith("po"):
        print("Unsupported, use 'get runtimes' instead")
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
        run_db = get_run_db()
        if name:
            run = run_db.read_run(name, project=project)
            print(dict_to_yaml(run))
            return

        runs = run_db.list_runs(uid=uid, project=project, labels=selector)
        df = runs.to_df()[
            ["name", "uid", "iter", "start", "state", "parameters", "results"]
        ]
        # df['uid'] = df['uid'].apply(lambda x: f'..{x[-6:]}')
        df["start"] = df["start"].apply(time_str)
        df["parameters"] = df["parameters"].apply(dict_to_str)
        df["results"] = df["results"].apply(dict_to_str)
        print(tabulate(df, headers="keys"))

    elif kind.startswith("art"):
        run_db = get_run_db()
        artifacts = run_db.list_artifacts(
            name, project=project, tag=tag, labels=selector
        )
        df = artifacts.to_df()[
            ["tree", "key", "iter", "kind", "path", "hash", "updated"]
        ]
        df["tree"] = df["tree"].apply(lambda x: f"..{x[-8:]}")
        df["hash"] = df["hash"].apply(lambda x: f"..{x[-6:]}")
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
            print("warning, reading workflows for all projects may take a long time !")
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
            "currently only get runs | runtimes | workflows | artifacts  | func [name] | runtime are supported"
        )


@main.command()
@click.option("--port", "-p", help="port to listen on", type=int)
@click.option("--dirpath", "-d", help="database directory (dirpath)")
@click.option("--dsn", "-s", help="database dsn, e.g. sqlite:///db/mlrun.db")
@click.option("--logs-path", "-l", help="logs directory path")
@click.option("--data-volume", "-v", help="path prefix to the location of artifacts")
@click.option("--verbose", is_flag=True, help="verbose log")
@click.option("--background", "-b", is_flag=True, help="run in background process")
@click.option("--artifact-path", "-a", help="default artifact path")
@click.option(
    "--update-env",
    default="",
    is_flag=False,
    flag_value=mlrun.config.default_env_file,
    help=f"update the specified mlrun .env file (if TEXT not provided defaults to {mlrun.config.default_env_file})",
)
def db(
    port,
    dirpath,
    dsn,
    logs_path,
    data_volume,
    verbose,
    background,
    artifact_path,
    update_env,
):
    """Run HTTP api/database server"""
    env = environ.copy()
    # ignore client side .env file (so import mlrun in server will not try to connect to local/remote DB)
    env["MLRUN_IGNORE_ENV_FILE"] = "true"
    env["MLRUN_DBPATH"] = ""

    if port is not None:
        env["MLRUN_httpdb__port"] = str(port)
    if dirpath is not None:
        env["MLRUN_httpdb__dirpath"] = dirpath
    if dsn is not None:
        if dsn.startswith("sqlite://") and "check_same_thread=" not in dsn:
            dsn += "?check_same_thread=false"
        env["MLRUN_HTTPDB__DSN"] = dsn
    if logs_path is not None:
        env["MLRUN_HTTPDB__LOGS_PATH"] = logs_path
    if data_volume is not None:
        env["MLRUN_HTTPDB__DATA_VOLUME"] = data_volume
    if verbose:
        env["MLRUN_LOG_LEVEL"] = "DEBUG"
    if artifact_path or "MLRUN_ARTIFACT_PATH" not in env:
        if not artifact_path:
            artifact_path = (
                env.get("MLRUN_HTTPDB__DATA_VOLUME", "./artifacts").rstrip("/")
                + "/{{project}}"
            )
        env["MLRUN_ARTIFACT_PATH"] = path.realpath(path.expanduser(artifact_path))

    env["MLRUN_IS_API_SERVER"] = "true"

    # create the DB dir if needed
    dsn = dsn or mlconf.httpdb.dsn
    if dsn and dsn.startswith("sqlite:///"):
        parsed = urlparse(dsn)
        p = pathlib.Path(parsed.path[1:]).parent
        p.mkdir(parents=True, exist_ok=True)

    cmd = [executable, "-m", "mlrun.api.main"]
    pid = None
    if background:
        print("Starting MLRun API service in the background...")
        child = Popen(
            cmd,
            env=env,
            stdout=open("mlrun-stdout.log", "w"),
            stderr=open("mlrun-stderr.log", "w"),
            start_new_session=True,
        )
        pid = child.pid
        print(
            f"background pid: {pid}, logs written to mlrun-stdout.log and mlrun-stderr.log, use:\n"
            f"`kill {pid}` (linux/mac) or `taskkill /pid {pid} /t /f` (windows), to kill the mlrun service process"
        )
    else:
        child = Popen(cmd, env=env)
        returncode = child.wait()
        if returncode != 0:
            raise SystemExit(returncode)
    if update_env:
        # update mlrun client env file with the API path, so client will use the new DB
        # update and PID, allow killing the correct process in a config script
        filename = path.expanduser(update_env)
        dotenv.set_key(
            filename, "MLRUN_DBPATH", f"http://localhost:{port or 8080}", quote_mode=""
        )
        dotenv.set_key(filename, "MLRUN_MOCK_NUCLIO_DEPLOYMENT", "auto", quote_mode="")
        if pid:
            dotenv.set_key(filename, "MLRUN_SERVICE_PID", str(pid), quote_mode="")
        print(f"updated configuration in {update_env} .env file")


@main.command()
def version():
    """get mlrun version"""
    print(f"MLRun version: {str(Version().get())}")


@main.command()
@click.argument("uid", type=str)
@click.option("--project", "-p", help="project name")
@click.option("--offset", type=int, default=0, help="byte offset")
@click.option("--db", help="api and db service path/url")
@click.option("--watch", "-w", is_flag=True, help="watch/follow log")
def logs(uid, project, offset, db, watch):
    """Get or watch task logs"""
    mldb = get_run_db(db or mlconf.dbpath)
    if mldb.kind == "http":
        state, _ = mldb.watch_log(uid, project, watch=watch, offset=offset)
    else:
        state, text = mldb.get_log(uid, project, offset=offset)
        if text:
            print(text.decode())

    if state:
        print(f"final state: {state}")


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
    "--ensure-project",
    is_flag=True,
    help="ensure the project exists, if not, create project",
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
# TODO: Remove in 1.6.0 --overwrite-schedule and -os, keep --override-workflow and -ow
@click.option(
    "--override-workflow",
    "--overwrite-schedule",
    "-ow",
    "-os",
    "override_workflow",
    is_flag=True,
    help="Override a schedule when submitting a new one with the same name.",
)
@click.option(
    "--save-secrets",
    is_flag=True,
    help="Store the project secrets as k8s secrets",
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
    ensure_project,
    schedule,
    override_workflow,
    save_secrets,
):
    """load and/or run a project"""
    if env_file:
        mlrun.set_env_from_file(env_file)

    if db:
        mlconf.dbpath = db

    proj = load_project(
        context, url, name, init_git=init_git, clone=clone, save=ensure_project
    )
    url_str = " from " + url if url else ""
    print(f"Loading project {proj.name}{url_str} into {context}:\n")

    if is_relative_path(artifact_path):
        artifact_path = path.abspath(artifact_path)
    if param:
        proj.spec.params = fill_params(param, proj.spec.params)
    if git_repo:
        proj.spec.params["git_repo"] = git_repo
    if git_issue:
        proj.spec.params["git_issue"] = git_issue
    commit = (
        proj.get_param("commit_id")
        or environ.get("GITHUB_SHA")
        or environ.get("CI_COMMIT_SHA")
    )
    if commit:
        proj.spec.params["commit_id"] = commit
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

        print(f"running workflow {run} file: {workflow_path}")
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
                override=override_workflow,
            )

        except Exception as exc:
            print(traceback.format_exc())
            message = f"failed to run pipeline, {err_to_str(exc)}"
            proj.notifiers.push(message, "error")
            exit(1)

    elif sync:
        print("saving project functions to db ..")
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
            print(f"error, env file {env_file} does not exist")
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
            print(f"updating configuration in .env file {filename}")
        env_dict = {
            "MLRUN_DBPATH": api,
            "MLRUN_ARTIFACT_PATH": artifact_path,
            "V3IO_USERNAME": username,
            "V3IO_ACCESS_KEY": access_key,
        }
        for key, value in env_dict.items():
            if value:
                dotenv.set_key(filename, key, value, quote_mode="")
        if env_vars:
            for key, value in list2dict(env_vars).items():
                dotenv.set_key(filename, key, value, quote_mode="")
        if env_file:
            # if its not the default file print the usage details
            print(
                f"to use the {env_file} .env file add the following to your development environment:\n"
                f"MLRUN_ENV_FILE={env_file}"
            )

    elif op == "clear":
        filename = path.expanduser(env_file or mlrun.config.default_env_file)
        if not path.isfile(filename):
            print(f".env file {filename} not found")
        else:
            print(f"deleting .env file {filename}")
            remove(filename)

    else:
        print(f"Error, unsupported config option {op}")


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


if __name__ == "__main__":
    main()
