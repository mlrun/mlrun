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
from os import environ, path
from pprint import pprint
from subprocess import Popen
from sys import executable

import click
import yaml
from tabulate import tabulate

import mlrun

from .builder import upload_tarball
from .config import config as mlconf
from .db import get_run_db
from .k8s_utils import K8sHelper
from .model import RunTemplate
from .platforms import auto_mount as auto_mount_modifier
from .projects import load_project
from .run import get_object, import_function, import_function_to_dict, new_function
from .runtimes import RemoteRuntime, RunError, RuntimeKinds, ServingRuntime
from .secrets import SecretsStore
from .utils import (
    dict_to_yaml,
    get_in,
    list2dict,
    logger,
    parse_versioned_object_uri,
    run_keys,
    update_in,
)
from .utils.version import Version


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("url", type=str, required=False)
@click.option(
    "--param",
    "-p",
    default="",
    multiple=True,
    help="parameter name and value tuples, e.g. -p x=37 -p y='text'",
)
@click.option("--inputs", "-i", multiple=True, help="input artifact")
@click.option("--outputs", "-o", multiple=True, help="output artifact/result for kfp")
@click.option("--in-path", help="default input path/url (prefix) for artifact")
@click.option("--out-path", help="default output path/url (prefix) for artifact")
@click.option(
    "--secrets", "-s", multiple=True, help="secrets file=<filename> or env=ENV_KEY1,.."
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
    default="",
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
    "--hyper-param-options", default="", help="hyperparam options json string",
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
@click.argument("run_args", nargs=-1, type=click.UNPROCESSED)
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
    run_args,
):
    """Execute a task and inject parameters."""

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

    if func_url or kind or image:
        if func_url:
            runtime = func_url_to_runtime(func_url)
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
        if suffix != ".py" and mode != "pass" and url_file != "{codefile}":
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
        resp = fn.run(runobj, watch=watch, schedule=schedule, local=local)
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print(f"runtime error: {err}")
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("func_url", type=str, required=False)
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
    default="",
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
):
    """Build a container image from code and requirements."""

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
    elif func_url.startswith("db://"):
        func_url = func_url[5:]
        func = import_function(func_url)
    elif func_url:
        func_url = "function.yaml" if func_url == "." else func_url
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

    if hasattr(func, "deploy"):
        logger.info("remote deployment started")
        try:
            func.deploy(
                with_mlrun=with_mlrun, watch=not silent, is_kfp=kfp, skip_deployed=skip
            )
        except Exception as err:
            print(f"deploy error, {err}")
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
@click.argument("spec", type=str, required=False)
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
def deploy(spec, source, func_url, dashboard, project, model, tag, kind, env, verbose):
    """Deploy model or function"""
    if func_url:
        runtime = func_url_to_runtime(func_url)
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
        print(f"deploy error: {err}")
        exit(1)

    print(f"function deployed, address={addr}")
    with open("/tmp/output", "w") as fp:
        fp.write(addr)
    with open("/tmp/name", "w") as fp:
        fp.write(function.status.nuclio_name)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pod", type=str)
@click.option("--namespace", "-n", help="kubernetes namespace")
@click.option(
    "--timeout", "-t", default=600, show_default=True, help="timeout in seconds"
)
def watch(pod, namespace, timeout):
    """Read current or previous task (pod) logs."""
    k8s = K8sHelper(namespace)
    status = k8s.watch(pod, namespace, timeout)
    print(f"Pod {pod} last status is: {status}")


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("kind", type=str)
@click.argument("name", type=str, default="", required=False)
@click.option("--selector", "-s", default="", help="label selector")
@click.option("--namespace", "-n", help="kubernetes namespace")
@click.option("--uid", help="unique ID")
@click.option("--project", "-p", help="project name")
@click.option("--tag", "-t", default="", help="artifact/function tag")
@click.option("--db", help="db path/url")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def get(kind, name, selector, namespace, uid, project, tag, db, extra_args):
    """List/get one or more object per kind/class."""

    if db:
        mlconf.dbpath = db

    if kind.startswith("po"):
        k8s = K8sHelper(namespace)
        if name:
            resp = k8s.get_pod(name, namespace)
            print(resp)
            return

        items = k8s.list_pods(namespace, selector)
        print(f"{'state':10} {'started':16} {'type':8} name")
        for i in items:
            task = i.metadata.labels.get("mlrun/class", "")
            if task:
                name = i.metadata.name
                state = i.status.phase
                start = ""
                if i.status.start_time:
                    start = i.status.start_time.strftime("%b %d %H:%M:%S")
                print(f"{state:10} {start:16} {task:8} {name}")
    elif kind.startswith("runtime"):
        run_db = get_run_db(db or mlconf.dbpath)
        if name:
            # the runtime identifier is its kind
            runtime = run_db.list_runtime_resources(kind=name, label_selector=selector)
            print(dict_to_yaml(runtime.dict()))
            return
        runtimes = run_db.list_runtime_resources(label_selector=selector)
        print(dict_to_yaml(runtimes.dict()))
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
    else:
        print(
            "currently only get pods | runs | artifacts | func [name] | runtime are supported"
        )


@main.command()
@click.option("--port", "-p", help="port to listen on", type=int)
@click.option("--dirpath", "-d", help="database directory (dirpath)")
def db(port, dirpath):
    """Run HTTP api/database server"""
    env = environ.copy()
    if port is not None:
        env["MLRUN_httpdb__port"] = str(port)
    if dirpath is not None:
        env["MLRUN_httpdb__dirpath"] = dirpath

    cmd = [executable, "-m", "mlrun.api.main"]
    child = Popen(cmd, env=env)
    returncode = child.wait()
    if returncode != 0:
        raise SystemExit(returncode)


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
        state = mldb.watch_log(uid, project, watch=watch, offset=offset)
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
    default="",
    multiple=True,
    help="Kubeflow pipeline arguments name and value tuples (with -r flag), e.g. -a x=6",
)
@click.option("--artifact-path", "-p", help="output artifacts path")
@click.option(
    "--param",
    "-x",
    default="",
    multiple=True,
    help="mlrun project parameter name and value tuples, e.g. -p x=37 -p y='text'",
)
@click.option(
    "--secrets", "-s", multiple=True, help="secrets file=<filename> or env=ENV_KEY1,.."
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
):
    """load and/or run a project"""
    if db:
        mlconf.dbpath = db

    proj = load_project(context, url, name, init_git=init_git, clone=clone)
    url_str = " from " + url if url else ""
    print(f"Loading project {proj.name}{url_str} into {context}:\n")

    if artifact_path and not ("://" in artifact_path or artifact_path.startswith("/")):
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
        proj._secrets = SecretsStore.from_list(secrets)
    print(proj.to_yaml())

    if run:
        workflow_path = None
        if run.endswith(".py"):
            workflow_path = run
            run = None

        args = None
        if arguments:
            args = fill_params(arguments)

        print(f"running workflow {run} file: {workflow_path}")
        message = run_result = ""
        had_error = False
        gitops = (
            git_issue
            or environ.get("GITHUB_EVENT_PATH")
            or environ.get("CI_MERGE_REQUEST_IID")
        )
        if gitops:
            proj.notifiers.git_comment(
                git_repo, git_issue, token=proj.get_secret("GITHUB_TOKEN")
            )
        try:
            run_result = proj.run(
                run,
                workflow_path,
                arguments=args,
                artifact_path=artifact_path,
                namespace=namespace,
                sync=sync,
                dirty=dirty,
                workflow_handler=handler,
                engine=engine,
                local=local,
            )
            print(f"run id: {run_result.run_id}")
        except Exception as exc:
            print(traceback.format_exc())
            message = f"failed to run pipeline, {exc}"
            had_error = True
            print(message)

        if had_error:
            proj.notifiers.push(message)
        if had_error:
            exit(1)

        if watch and run_result and run_result.workflow.engine == "kfp":
            proj.get_run_status(run_result)

    elif sync:
        print("saving project functions to db ..")
        proj.sync_functions(save=True)


def validate_kind(ctx, param, value):
    possible_kinds = RuntimeKinds.runtime_with_handlers()
    if value is not None and value not in possible_kinds:
        raise click.BadParameter(
            f"kind must be one of {possible_kinds}", ctx=ctx, param=param
        )
    return value


@main.command()
@click.argument("kind", callback=validate_kind, default=None, required=False)
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
def show_config():
    """Show configuration & exit"""
    print(mlconf.dump_yaml())


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


def func_url_to_runtime(func_url):
    try:
        if func_url.startswith("db://"):
            func_url = func_url[5:]
            project_instance, name, tag, hash_key = parse_versioned_object_uri(func_url)
            run_db = get_run_db(mlconf.dbpath)
            runtime = run_db.get_function(name, project_instance, tag, hash_key)
        else:
            func_url = "function.yaml" if func_url == "." else func_url
            runtime = import_function_to_dict(func_url, {})
    except Exception as exc:
        logger.error(f"function {func_url} not found, {exc}")
        return None

    if not runtime:
        logger.error(f"function {func_url} not found or is null")
        return None

    return runtime


if __name__ == "__main__":
    main()
