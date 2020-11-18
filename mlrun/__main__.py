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

from .builder import upload_tarball
from .config import config as mlconf
from .db import get_run_db
from .k8s_utils import K8sHelper
from .model import RunTemplate
from .projects import load_project
from .run import new_function, import_function_to_dict, import_function, get_object
from .runtimes import RemoteRuntime, RunError, RuntimeKinds, ServingRuntime
from .secrets import SecretsStore
from .utils import (
    list2dict,
    logger,
    run_keys,
    update_in,
    get_in,
    parse_function_uri,
    dict_to_yaml,
    pr_comment,
    RunNotifications,
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
    "--tuning-strategy",
    default="",
    help="hyperparam tuning strategy list | grid | random",
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
@click.option("--mode", help="special run mode noctx | pass")
@click.option("--schedule", help="cron schedule")
@click.option("--from-env", is_flag=True, help="read the spec from the env var")
@click.option("--dump", is_flag=True, help="dump run results as YAML")
@click.option("--image", default="", help="container image")
@click.option("--workdir", default="", help="run working directory")
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
    tuning_strategy,
    func_url,
    task,
    handler,
    mode,
    schedule,
    from_env,
    dump,
    image,
    workdir,
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

    if db:
        mlconf.dbpath = db

    if func_url:
        runtime = func_url_to_runtime(func_url)
        if runtime is None:
            exit(1)
        kind = get_in(runtime, "kind", "")
        if kind not in ["", "local", "dask"] and url:
            if path.isfile(url) and url.endswith(".py"):
                with open(url) as fp:
                    body = fp.read()
                based = b64encode(body.encode("utf-8")).decode("utf-8")
                logger.info("packing code at {}".format(url))
                update_in(runtime, "spec.build.functionSourceCode", based)
                url = ""
                update_in(runtime, "spec.command", "")
    elif runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print("runtime parameter must be a dict, not {}".format(type(runtime)))
            exit(1)
    else:
        runtime = {}

    code = environ.get("MLRUN_EXEC_CODE")
    if get_in(runtime, "kind", "") == "dask":
        code = get_in(runtime, "spec.build.functionSourceCode", code)
    if from_env and code:
        code = b64decode(code).decode("utf-8")
        if kfp:
            print("code:\n{}\n".format(code))
        with open("main.py", "w") as fp:
            fp.write(code)
        url = url or "main.py"

    if url:
        update_in(runtime, "spec.command", url)
    if run_args:
        update_in(runtime, "spec.args", list(run_args))
    if image:
        update_in(runtime, "spec.image", image)
    set_item(runobj.spec, handler, "handler")
    set_item(runobj.spec, param, "parameters", fill_params(param))
    set_item(runobj.spec, hyperparam, "hyperparams", fill_params(hyperparam))
    set_item(runobj.spec, param_file, "param_file")
    set_item(runobj.spec, tuning_strategy, "tuning_strategy")
    set_item(runobj.spec, selector, "selector")

    set_item(runobj.spec, inputs, run_keys.inputs, list2dict(inputs))
    set_item(runobj.spec, in_path, run_keys.input_path)
    set_item(runobj.spec, out_path, run_keys.output_path)
    set_item(runobj.spec, outputs, run_keys.outputs, list(outputs))
    set_item(
        runobj.spec, secrets, run_keys.secrets, line2keylist(secrets, "kind", "source")
    )
    set_item(runobj.spec, verbose, "verbose")
    set_item(runobj.spec, scrape_metrics, "scrape_metrics")

    if kfp or runobj.spec.verbose or verbose:
        print("MLRun version: {}".format(str(Version().get())))
        print("Runtime:")
        pprint(runtime)
        print("Run:")
        pprint(runobj.to_dict())

    try:
        update_in(runtime, "metadata.name", name, replace=False)
        fn = new_function(runtime=runtime, kfp=kfp, mode=mode)
        if workdir:
            fn.spec.workdir = workdir
        fn.is_child = from_env and not kfp
        resp = fn.run(runobj, watch=watch, schedule=schedule)
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print("runtime error: {}".format(err))
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
            print("runtime parameter must be a dict, not {}".format(type(runtime)))
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
            print("source file doesnt exist ({})".format(source))
            exit(1)
        with open(source) as fp:
            body = fp.read()
        based = b64encode(body.encode("utf-8")).decode("utf-8")
        logger.info("packing code at {}".format(source))
        b.functionSourceCode = based
        func.spec.command = ""
    else:
        b.source = source or b.source
        # todo: upload stuff

    archive = archive or mlconf.default_archive
    if archive:
        src = b.source or "./"
        logger.info("uploading data from {} to {}".format(src, archive))
        target = archive if archive.endswith("/") else archive + "/"
        target += "src-{}-{}-{}.tar.gz".format(
            meta.project, meta.name, meta.tag or "latest"
        )
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
            print("deploy error, {}".format(err))
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

        print("function built, state={} image={}".format(state, image))
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
        print("runtime parameter must be a dict, not {}".format(type(runtime)))
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
        print("deploy error: {}".format(err))
        exit(1)

    print("function deployed, address={}".format(addr))
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
    print("Pod {} last status is: {}".format(pod, status))


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
        print("{:10} {:16} {:8} {}".format("state", "started", "type", "name"))
        for i in items:
            task = i.metadata.labels.get("mlrun/class", "")
            if task:
                name = i.metadata.name
                state = i.status.phase
                start = ""
                if i.status.start_time:
                    start = i.status.start_time.strftime("%b %d %H:%M:%S")
                print("{:10} {:16} {:8} {}".format(state, start, task, name))
    elif kind.startswith("runtime"):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        if name:
            # the runtime identifier is its kind
            runtime = mldb.get_runtime(kind=name, label_selector=selector)
            print(dict_to_yaml(runtime))
            return
        runtimes = mldb.list_runtimes(label_selector=selector)
        print(dict_to_yaml(runtimes))
    elif kind.startswith("run"):
        mldb = get_run_db().connect()
        if name:
            run = mldb.read_run(name, project=project)
            print(dict_to_yaml(run))
            return

        runs = mldb.list_runs(uid=uid, project=project, labels=selector)
        df = runs.to_df()[
            ["name", "uid", "iter", "start", "state", "parameters", "results"]
        ]
        # df['uid'] = df['uid'].apply(lambda x: '..{}'.format(x[-6:]))
        df["start"] = df["start"].apply(time_str)
        df["parameters"] = df["parameters"].apply(dict_to_str)
        df["results"] = df["results"].apply(dict_to_str)
        print(tabulate(df, headers="keys"))

    elif kind.startswith("art"):
        mldb = get_run_db().connect()
        artifacts = mldb.list_artifacts(name, project=project, tag=tag, labels=selector)
        df = artifacts.to_df()[
            ["tree", "key", "iter", "kind", "path", "hash", "updated"]
        ]
        df["tree"] = df["tree"].apply(lambda x: "..{}".format(x[-8:]))
        df["hash"] = df["hash"].apply(lambda x: "..{}".format(x[-6:]))
        print(tabulate(df, headers="keys"))

    elif kind.startswith("func"):
        mldb = get_run_db().connect()
        if name:
            f = mldb.get_function(name, project=project, tag=tag)
            print(dict_to_yaml(f))
            return

        functions = mldb.list_functions(name, project=project, labels=selector)
        lines = []
        headers = ["kind", "state", "name:tag", "hash"]
        for f in functions:
            line = [
                get_in(f, "kind", ""),
                get_in(f, "status.state", ""),
                "{}:{}".format(
                    get_in(f, "metadata.name"), get_in(f, "metadata.tag", "")
                ),
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
    print("MLRun version: {}".format(str(Version().get())))


@main.command()
@click.argument("uid", type=str)
@click.option("--project", "-p", help="project name")
@click.option("--offset", type=int, default=0, help="byte offset")
@click.option("--db", help="api and db service path/url")
@click.option("--watch", "-w", is_flag=True, help="watch/follow log")
def logs(uid, project, offset, db, watch):
    """Get or watch task logs"""
    mldb = get_run_db(db or mlconf.dbpath).connect()
    if mldb.kind == "http":
        state = mldb.watch_log(uid, project, watch=watch, offset=offset)
    else:
        state, text = mldb.get_log(uid, project, offset=offset)
        if text:
            print(text.decode())

    if state:
        print("final state: {}".format(state))


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
    help="Kubeflow pipeline arguments name and value tuples, e.g. -a x=6",
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
    "--clone", "-c", is_flag=True, help="force override/clone the context dir"
)
@click.option("--sync", is_flag=True, help="sync functions into db")
@click.option(
    "--watch", "-w", is_flag=True, help="wait for pipeline completion (with -r flag)"
)
@click.option("--dirty", "-d", is_flag=True, help="allow git with uncommited changes")
@click.option("--git-repo", help="git repo (org/repo) for git comments")
@click.option(
    "--git-issue", type=int, default=None, help="git issue number for git comments"
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
):
    """load and/or run a project"""
    if db:
        mlconf.dbpath = db

    proj = load_project(context, url, name, init_git=init_git, clone=clone)
    print(
        "Loading project {}{} into {}:\n".format(
            proj.name, " from " + url if url else "", context
        )
    )

    if artifact_path and not ("://" in artifact_path or artifact_path.startswith("/")):
        artifact_path = path.abspath(artifact_path)
    if param:
        proj.params = fill_params(param, proj.params)
    if git_repo:
        proj.params["git_repo"] = git_repo
    if git_issue:
        proj.params["git_issue"] = git_issue
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

        print("running workflow {} file: {}".format(run, workflow_path))
        message = run = ""
        had_error = False
        try:
            run = proj.run(
                run,
                workflow_path,
                arguments=args,
                artifact_path=artifact_path,
                namespace=namespace,
                sync=sync,
                dirty=dirty,
            )
        except Exception as e:
            print(traceback.format_exc())
            message = "failed to run pipeline, {}".format(e)
            had_error = True
            print(message)
        print("run id: {}".format(run))

        gitops = git_repo and git_issue
        if gitops:
            if not had_error:
                message = "Pipeline started id={}".format(run)
                if proj.params and "commit" in proj.params:
                    message += ", commit={}".format(proj.params["commit"])
                if mlconf.ui_url:
                    temp = (
                        '<div><a href="{}/projects/{}/jobs" target='
                        + ' "_blank">click here to check progress</a></div>'
                    )
                    message += temp.format(mlconf.ui_url, proj.name)
            pr_comment(
                git_repo, git_issue, message, token=proj.get_secret("GITHUB_TOKEN")
            )

        if had_error:
            exit(1)

        if watch:
            n = RunNotifications(with_slack=True).print()
            if gitops:
                n.git_comment(
                    git_repo, git_issue, token=proj.get_secret("GITHUB_TOKEN")
                )
            proj.get_run_status(run, notifiers=n)

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
    default=mlconf.runtime_resources_deletion_grace_period,
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
        mlrun clean dask 15d04c19c2194c0a8efb26ea3017254b
    """
    mldb = get_run_db(api or mlconf.dbpath).connect()
    if kind:
        if object_id:
            mldb.delete_runtime_object(
                kind=kind,
                object_id=object_id,
                label_selector=label_selector,
                force=force,
                grace_period=grace_period,
            )
        else:
            mldb.delete_runtime(
                kind=kind,
                label_selector=label_selector,
                force=force,
                grace_period=grace_period,
            )
    else:
        mldb.delete_runtimes(
            label_selector=label_selector, force=force, grace_period=grace_period
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
            raise ValueError("cannot find param key in line ({})".format(param))
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
            raise ValueError('cannot find "=" in line ({}={})'.format(keyname, valname))
        key, value = line[:i].strip(), line[i + 1 :].strip()
        if key is None:
            raise ValueError("cannot find key in line ({}={})".format(keyname, valname))
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
    return ",".join(["{}={}".format(k, v) for k, v in struct.items()])


def func_url_to_runtime(func_url):
    try:
        if func_url.startswith("db://"):
            func_url = func_url[5:]
            project, name, tag, hash_key = parse_function_uri(func_url)
            mldb = get_run_db(mlconf.dbpath).connect()
            runtime = mldb.get_function(name, project, tag, hash_key)
        else:
            func_url = "function.yaml" if func_url == "." else func_url
            runtime = import_function_to_dict(func_url, {})
    except Exception as e:
        logger.error("function {} not found, {}".format(func_url, e))
        return None

    if not runtime:
        logger.error("function {} not found or is null".format(func_url))
        return None

    return runtime


if __name__ == "__main__":
    main()
