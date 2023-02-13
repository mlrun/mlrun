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
import functools
import importlib.util as imputil
import inspect
import json
import os
import pathlib
import shutil
import socket
import tempfile
import time
import uuid
import warnings
from base64 import b64decode
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from os import environ, makedirs, path
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import cloudpickle
import nuclio
import numpy as np
import pandas as pd
import yaml
from kfp import Client

import mlrun.api.schemas
import mlrun.errors
import mlrun.utils.helpers
from mlrun.kfpops import format_summary_from_kfp_run, show_kfp_run

from .config import config as mlconf
from .datastore import DataItem, store_manager
from .db import get_or_set_dburl, get_run_db
from .execution import MLClientCtx
from .model import BaseMetadata, RunObject, RunTemplate
from .runtimes import (
    DaskCluster,
    HandlerRuntime,
    KubejobRuntime,
    LocalRuntime,
    MpiRuntimeV1,
    MpiRuntimeV1Alpha1,
    RemoteRuntime,
    RemoteSparkRuntime,
    RuntimeKinds,
    ServingRuntime,
    Spark2Runtime,
    Spark3Runtime,
    get_runtime_class,
)
from .runtimes.funcdoc import update_function_entry_points
from .runtimes.serving import serving_subkind
from .runtimes.utils import add_code_metadata, global_context
from .utils import (
    extend_hub_uri_if_needed,
    get_in,
    logger,
    new_pipe_metadata,
    parse_versioned_object_uri,
    retry_until_successful,
    run_keys,
    update_in,
)


class RunStatuses(object):
    succeeded = "Succeeded"
    failed = "Failed"
    skipped = "Skipped"
    error = "Error"
    running = "Running"

    @staticmethod
    def all():
        return [
            RunStatuses.succeeded,
            RunStatuses.failed,
            RunStatuses.skipped,
            RunStatuses.error,
            RunStatuses.running,
        ]

    @staticmethod
    def stable_statuses():
        return [
            RunStatuses.succeeded,
            RunStatuses.failed,
            RunStatuses.skipped,
            RunStatuses.error,
        ]

    @staticmethod
    def transient_statuses():
        return [
            status
            for status in RunStatuses.all()
            if status not in RunStatuses.stable_statuses()
        ]


def run_local(
    task=None,
    command="",
    name: str = "",
    args: list = None,
    workdir=None,
    project: str = "",
    tag: str = "",
    secrets=None,
    handler=None,
    params: dict = None,
    inputs: dict = None,
    artifact_path: str = "",
    mode: str = None,
    allow_empty_resources=None,
):
    """Run a task on function/code (.py, .ipynb or .yaml) locally,

    example::

        # define a task
        task = new_task(params={'p1': 8}, out_path=out_path)
        # run
        run = run_local(spec, command='src/training.py', workdir='src')

    or specify base task parameters (handler, params, ..) in the call::

        run = run_local(handler=my_function, params={'x': 5})

    :param task:     task template object or dict (see RunTemplate)
    :param command:  command/url/function
    :param name:     ad hook function name
    :param args:     command line arguments (override the ones in command)
    :param workdir:  working dir to exec in
    :param project:  function project (none for 'default')
    :param tag:      function version tag (none for 'latest')
    :param secrets:  secrets dict if the function source is remote (s3, v3io, ..)

    :param handler:  pointer or name of a function handler
    :param params:   input parameters (dict)
    :param inputs:   input objects (dict of key: path)
    :param artifact_path: default artifact output path
    :param mode:    Runtime mode for more details head to `mlrun.new_function`
    :param allow_empty_resources:   Allow passing non materialized set/vector as input to jobs
                                    (allows to have function which don't depend on having targets,
                                    e.g a function which accepts a feature vector uri and generate
                                     the offline vector e.g. parquet_ for it if it doesn't exist)

    :return: run object
    """

    function_name = name
    if command and isinstance(command, str):
        sp = command.split()
        command = sp[0]
        if len(sp) > 1:
            args = args or []
            args = sp[1:] + args
        function_name = function_name or pathlib.Path(command).stem

    meta = BaseMetadata(function_name, project=project, tag=tag)
    command, runtime = load_func_code(command, workdir, secrets=secrets, name=name)

    if runtime:
        if task:
            handler = handler or task.spec.handler
        handler = handler or runtime.spec.default_handler or ""
        meta = runtime.metadata.copy()
        meta.project = project or meta.project
        meta.tag = tag or meta.tag

    # if the handler has module prefix force "local" (vs "handler") runtime
    kind = "local" if isinstance(handler, str) and "." in handler else ""
    fn = new_function(meta.name, command=command, args=args, mode=mode, kind=kind)
    fn.metadata = meta
    setattr(fn, "_is_run_local", True)
    if workdir:
        fn.spec.workdir = str(workdir)
    fn.spec.allow_empty_resources = allow_empty_resources
    if runtime:
        # copy the code/base-spec to the local function (for the UI and code logging)
        fn.spec.description = runtime.spec.description
        fn.spec.build = runtime.spec.build
    return fn.run(
        task,
        name=name,
        handler=handler,
        params=params,
        inputs=inputs,
        artifact_path=artifact_path,
    )


def function_to_module(code="", workdir=None, secrets=None, silent=False):
    """Load code, notebook or mlrun function as .py module
    this function can import a local/remote py file or notebook
    or load an mlrun function object as a module, you can use this
    from your code, notebook, or another function (for common libs)

    Note: the function may have package requirements which must be satisfied

    example::

        mod = mlrun.function_to_module('./examples/training.py')
        task = mlrun.new_task(inputs={'infile.txt': '../examples/infile.txt'})
        context = mlrun.get_or_create_ctx('myfunc', spec=task)
        mod.my_job(context, p1=1, p2='x')
        print(context.to_yaml())

        fn = mlrun.import_function('hub://open_archive')
        mod = mlrun.function_to_module(fn)
        data = mlrun.run.get_dataitem("https://fpsignals-public.s3.amazonaws.com/catsndogs.tar.gz")
        context = mlrun.get_or_create_ctx('myfunc')
        mod.open_archive(context, archive_url=data)
        print(context.to_yaml())

    :param code:    path/url to function (.py or .ipynb or .yaml)
                    OR function object
    :param workdir: code workdir
    :param secrets: secrets needed to access the URL (e.g.s3, v3io, ..)
    :param silent:  do not raise on errors

    :returns: python module
    """
    command, _ = load_func_code(code, workdir, secrets=secrets)
    if not command:
        if silent:
            return None
        raise ValueError("nothing to run, specify command or function")

    command = os.path.join(workdir or "", command)
    path = Path(command)
    mod_name = path.name
    if path.suffix:
        mod_name = mod_name[: -len(path.suffix)]
    spec = imputil.spec_from_file_location(mod_name, command)
    if spec is None:
        raise OSError(f"cannot import from {command!r}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def load_func_code(command="", workdir=None, secrets=None, name="name"):
    is_obj = hasattr(command, "to_dict")
    suffix = "" if is_obj else Path(command).suffix
    runtime = None
    if is_obj or suffix == ".yaml":
        is_remote = False
        if is_obj:
            runtime = command
        else:
            is_remote = "://" in command
            data = get_object(command, secrets)
            runtime = yaml.load(data, Loader=yaml.FullLoader)
            runtime = new_function(runtime=runtime)

        command = runtime.spec.command or ""
        code = runtime.spec.build.functionSourceCode
        origin_filename = runtime.spec.build.origin_filename
        kind = runtime.kind or ""
        if kind in RuntimeKinds.nuclio_runtimes():
            code = get_in(runtime.spec.base_spec, "spec.build.functionSourceCode", code)
        if code:
            if (
                origin_filename
                and origin_filename.endswith(".py")
                and path.isfile(origin_filename)
            ):
                command = origin_filename
            else:
                suffix = ".py"
                if origin_filename:
                    suffix = f"-{pathlib.Path(origin_filename).stem}.py"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, mode="w", delete=False
                ) as temp_file:
                    code = b64decode(code).decode("utf-8")
                    command = temp_file.name
                    temp_file.write(code)

        elif command and not is_remote:
            file_path = path.join(workdir or "", command)
            if not path.isfile(file_path):
                raise OSError(f"command file {file_path} not found")

        else:
            logger.warn("run command, file or code were not specified")

    elif command == "":
        pass

    elif suffix == ".ipynb":
        temp_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        code_to_function(
            name, filename=command, kind="local", code_output=temp_file.name
        )
        command = temp_file.name

    elif suffix == ".py":
        if "://" in command:
            temp_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
            download_object(command, temp_file.name, secrets)
            command = temp_file.name

    else:
        raise ValueError(f"unsupported suffix: {suffix}")

    return command, runtime


def get_or_create_ctx(
    name: str,
    event=None,
    spec=None,
    with_env: bool = True,
    rundb: str = "",
    project: str = "",
    upload_artifacts=False,
):
    """called from within the user program to obtain a run context

    the run context is an interface for receiving parameters, data and logging
    run results, the run context is read from the event, spec, or environment
    (in that order), user can also work without a context (local defaults mode)

    all results are automatically stored in the "rundb" or artifact store,
    the path to the rundb can be specified in the call or obtained from env.

    :param name:     run name (will be overridden by context)
    :param event:    function (nuclio Event object)
    :param spec:     dictionary holding run spec
    :param with_env: look for context in environment vars, default True
    :param rundb:    path/url to the metadata and artifact database
    :param project:  project to initiate the context in (by default mlrun.mlctx.default_project)
    :param upload_artifacts:  when using local context (not as part of a job/run), upload artifacts to the
                              system default artifact path location

    :return: execution context

    Examples::

        # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
        context = get_or_create_ctx('train')

        # get parameters from the runtime context (or use defaults)
        p1 = context.get_param('p1', 1)
        p2 = context.get_param('p2', 'a-string')

        # access input metadata, values, files, and secrets (passwords)
        print(f'Run: {context.name} (uid={context.uid})')
        print(f'Params: p1={p1}, p2={p2}')
        print(f'accesskey = {context.get_secret("ACCESS_KEY")}')
        input_str = context.get_input('infile.txt').get()
        print(f'file: {input_str}')

        # RUN some useful code e.g. ML training, data prep, etc.

        # log scalar result values (job result metrics)
        context.log_result('accuracy', p1 * 2)
        context.log_result('loss', p1 * 3)
        context.set_label('framework', 'sklearn')

        # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
        context.log_artifact('model.txt', body=b'abc is 123', labels={'framework': 'xgboost'})
        context.log_artifact('results.html', body=b'<b> Some HTML <b>', viewer='web-app')

    """

    if global_context.get() and not spec and not event:
        return global_context.get()

    newspec = {}
    config = environ.get("MLRUN_EXEC_CONFIG")
    if event:
        newspec = event.body

    elif spec:
        newspec = deepcopy(spec)

    elif with_env and config:
        newspec = config

    if isinstance(newspec, (RunObject, RunTemplate)):
        newspec = newspec.to_dict()

    if newspec and not isinstance(newspec, dict):
        newspec = json.loads(newspec)

    if not newspec:
        newspec = {}
        if upload_artifacts:
            artifact_path = mlrun.utils.helpers.fill_artifact_path_template(
                mlconf.artifact_path, project or mlconf.default_project
            )
            update_in(newspec, ["spec", run_keys.output_path], artifact_path)

    newspec.setdefault("metadata", {})
    update_in(newspec, "metadata.name", name, replace=False)
    autocommit = False
    tmp = environ.get("MLRUN_META_TMPFILE")
    out = rundb or mlconf.dbpath or environ.get("MLRUN_DBPATH")
    if out:
        autocommit = True
        logger.info(f"logging run results to: {out}")

    newspec["metadata"]["project"] = (
        newspec["metadata"].get("project") or project or mlconf.default_project
    )

    ctx = MLClientCtx.from_dict(
        newspec, rundb=out, autocommit=autocommit, tmp=tmp, host=socket.gethostname()
    )
    global_context.set(ctx)
    return ctx


def import_function(url="", secrets=None, db="", project=None, new_name=None):
    """Create function object from DB or local/remote YAML file

    Functions can be imported from function repositories (mlrun Function Hub (formerly Marketplace) or local db),
    or be read from a remote URL (http(s), s3, git, v3io, ..) containing the function YAML

    special URLs::

        function hub: hub://{name}[:{tag}]
        local mlrun db:       db://{project-name}/{name}[:{tag}]

    examples::

        function = mlrun.import_function("hub://auto_trainer")
        function = mlrun.import_function("./func.yaml")
        function = mlrun.import_function("https://raw.githubusercontent.com/org/repo/func.yaml")

    :param url: path/url to Function Hub, db or function YAML file
    :param secrets: optional, credentials dict for DB or URL (s3, v3io, ...)
    :param db: optional, mlrun api/db path
    :param project: optional, target project for the function
    :param new_name: optional, override the imported function name

    :returns: function object
    """
    is_hub_uri = False
    if url.startswith("db://"):
        url = url[5:]
        _project, name, tag, hash_key = parse_versioned_object_uri(url)
        db = get_run_db(db or get_or_set_dburl(), secrets=secrets)
        runtime = db.get_function(name, _project, tag, hash_key)
        if not runtime:
            raise KeyError(f"function {name}:{tag} not found in the DB")
    else:
        url, is_hub_uri = extend_hub_uri_if_needed(url)
        runtime = import_function_to_dict(url, secrets)
    function = new_function(runtime=runtime)
    project = project or mlrun.mlconf.default_project
    # When we're importing from the hub we want to assign to a target project, otherwise any store on it will
    # simply default to the default project
    if project and is_hub_uri:
        function.metadata.project = project
    if new_name:
        function.metadata.name = mlrun.utils.helpers.normalize_name(new_name)
    return function


def import_function_to_dict(url, secrets=None):
    """Load function spec from local/remote YAML file"""
    obj = get_object(url, secrets)
    runtime = yaml.load(obj, Loader=yaml.FullLoader)
    remote = "://" in url

    code = get_in(runtime, "spec.build.functionSourceCode")
    update_in(runtime, "metadata.build.code_origin", url)
    cmd = code_file = get_in(runtime, "spec.command", "")
    if " " in cmd:
        code_file = cmd[: cmd.find(" ")]
    if runtime["kind"] in ["", "local"]:
        if code:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False
            ) as temp_file:
                code = b64decode(code).decode("utf-8")
                update_in(runtime, "spec.command", temp_file.name)
                temp_file.write(code)
        elif remote and cmd:
            if cmd.startswith("/"):
                raise ValueError("exec path (spec.command) must be relative")
            url = url[: url.rfind("/") + 1] + code_file
            code = get_object(url, secrets)
            dir = path.dirname(code_file)
            if dir:
                makedirs(dir, exist_ok=True)
            with open(code_file, "wb") as fp:
                fp.write(code)
        elif cmd:
            if not path.isfile(code_file):
                # look for the file in a relative path to the yaml
                slash = url.rfind("/")
                if slash >= 0 and path.isfile(url[: url.rfind("/") + 1] + code_file):
                    raise ValueError(
                        f"exec file spec.command={code_file} is relative, change working dir"
                    )
                raise ValueError(f"no file in exec path (spec.command={code_file})")
        else:
            raise ValueError("command or code not specified in function spec")

    return runtime


def new_function(
    name: str = "",
    project: str = "",
    tag: str = "",
    kind: str = "",
    command: str = "",
    image: str = "",
    args: list = None,
    runtime=None,
    mode=None,
    handler: str = None,
    source: str = None,
    requirements: Union[str, List[str]] = None,
    kfp=None,
):
    """Create a new ML function from base properties

    example::

           # define a container based function (the `training.py` must exist in the container workdir)
           f = new_function(command='training.py -x {x}', image='myrepo/image:latest', kind='job')
           f.run(params={"x": 5})

           # define a container based function which reads its source from a git archive
           f = new_function(command='training.py -x {x}', image='myrepo/image:latest', kind='job',
                            source='git://github.com/mlrun/something.git')
           f.run(params={"x": 5})

           # define a local handler function (execute a local function handler)
           f = new_function().run(task, handler=myfunction)

    :param name:     function name
    :param project:  function project (none for 'default')
    :param tag:      function version tag (none for 'latest')

    :param kind:     runtime type (local, job, nuclio, spark, mpijob, dask, ..)
    :param command:  command/url + args (e.g.: training.py --verbose)
    :param image:    container image (start with '.' for default registry)
    :param args:     command line arguments (override the ones in command)
    :param runtime:  runtime (job, nuclio, spark, dask ..) object/dict
                     store runtime specific details and preferences
    :param mode:     runtime mode, "args" mode will push params into command template, example:
                      command=`mycode.py --x {xparam}` will substitute the `{xparam}` with the value of the xparam param
                     "pass" mode will run the command as is in the container (not wrapped by mlrun), the command can use
                      `{}` for parameters like in the "args" mode
    :param handler:  The default function handler to call for the job or nuclio function, in batch functions
                     (job, mpijob, ..) the handler can also be specified in the `.run()` command, when not specified
                     the entire file will be executed (as main).
                     for nuclio functions the handler is in the form of module:function, defaults to "main:handler"
    :param source:   valid path to git, zip, or tar file, e.g. `git://github.com/mlrun/something.git`,
                     `http://some/url/file.zip`
    :param requirements: list of python packages or pip requirements file path, defaults to None
    :param kfp:      reserved, flag indicating running within kubeflow pipeline

    :return: function object
    """
    # don't override given dict
    if runtime and isinstance(runtime, dict):
        runtime = deepcopy(runtime)
    kind, runtime = _process_runtime(command, runtime, kind)
    command = get_in(runtime, "spec.command", command)
    name = name or get_in(runtime, "metadata.name", "")

    if not kind and not command:
        runner = HandlerRuntime()
    else:
        if kind in ["", "local"] and command:
            runner = LocalRuntime.from_dict(runtime)
        elif kind in RuntimeKinds.all():
            runner = get_runtime_class(kind).from_dict(runtime)
        else:
            supported_runtimes = ",".join(RuntimeKinds.all())
            raise Exception(
                f"unsupported runtime ({kind}) or missing command, supported runtimes: {supported_runtimes}"
            )

    if not name:
        if command and kind not in [RuntimeKinds.remote]:
            name, _ = path.splitext(path.basename(command))
        else:
            name = "mlrun-" + uuid.uuid4().hex[0:6]

    # make sure function name is valid
    name = mlrun.utils.helpers.normalize_name(name)

    runner.metadata.name = name
    runner.metadata.project = (
        runner.metadata.project or project or mlconf.default_project
    )
    if tag:
        runner.metadata.tag = tag
    if image:
        if kind in ["", "handler", "local"]:
            raise ValueError(
                "image should only be set with containerized "
                "runtimes (job, mpijob, spark, ..), set kind=.."
            )
        runner.spec.image = image
    if args:
        runner.spec.args = args
    runner.kfp = kfp
    if mode:
        runner.spec.mode = mode
    if source:
        runner.spec.build.source = source
    if handler:
        if kind == RuntimeKinds.serving:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot set the handler for serving runtime"
            )
        elif kind in RuntimeKinds.nuclio_runtimes():
            runner.spec.function_handler = handler
        else:
            runner.spec.default_handler = handler

    if requirements:
        runner.with_requirements(requirements)
    runner.verify_base_image()
    return runner


def _process_runtime(command, runtime, kind):
    if runtime and hasattr(runtime, "to_dict"):
        runtime = runtime.to_dict()
    if runtime and isinstance(runtime, dict):
        kind = kind or runtime.get("kind", "")
        command = command or get_in(runtime, "spec.command", "")
    if "://" in command and command.startswith("http"):
        kind = kind or RuntimeKinds.remote
    if not runtime:
        runtime = {}
    update_in(runtime, "spec.command", command)
    runtime["kind"] = kind
    if kind != RuntimeKinds.remote:
        if command:
            update_in(runtime, "spec.command", command)
    else:
        update_in(runtime, "spec.function_kind", "mlrun")
    return kind, runtime


def code_to_function(
    name: str = "",
    project: str = "",
    tag: str = "",
    filename: str = "",
    handler: str = "",
    kind: str = "",
    image: str = None,
    code_output: str = "",
    embed_code: bool = True,
    description: str = "",
    requirements: Union[str, List[str]] = None,
    categories: List[str] = None,
    labels: Dict[str, str] = None,
    with_doc: bool = True,
    ignored_tags=None,
) -> Union[
    MpiRuntimeV1Alpha1,
    MpiRuntimeV1,
    RemoteRuntime,
    ServingRuntime,
    DaskCluster,
    KubejobRuntime,
    LocalRuntime,
    Spark2Runtime,
    Spark3Runtime,
    RemoteSparkRuntime,
]:
    """Convenience function to insert code and configure an mlrun runtime.

    Easiest way to construct a runtime type object. Provides the most often
    used configuration options for all runtimes as parameters.

    Instantiated runtimes are considered 'functions' in mlrun, but they are
    anything from nuclio functions to generic kubernetes pods to spark jobs.
    Functions are meant to be focused, and as such limited in scope and size.
    Typically a function can be expressed in a single python module with
    added support from custom docker images and commands for the environment.
    The returned runtime object can be further configured if more
    customization is required.

    One of the most important parameters is 'kind'. This is what is used to
    specify the chosen runtimes. The options are:

    - local: execute a local python or shell script
    - job: insert the code into a Kubernetes pod and execute it
    - nuclio: insert the code into a real-time serverless nuclio function
    - serving: insert code into orchestrated nuclio function(s) forming a DAG
    - dask: run the specified python code / script as Dask Distributed job
    - mpijob: run distributed Horovod jobs over the MPI job operator
    - spark: run distributed Spark job using Spark Kubernetes Operator
    - remote-spark: run distributed Spark job on remote Spark service

    Learn more about function runtimes here:
    https://docs.mlrun.org/en/latest/runtimes/functions.html#function-runtimes

    :param name:         function name, typically best to use hyphen-case
    :param project:      project used to namespace the function, defaults to 'default'
    :param tag:          function tag to track multiple versions of the same function, defaults to 'latest'
    :param filename:     path to .py/.ipynb file, defaults to current jupyter notebook
    :param handler:      The default function handler to call for the job or nuclio function, in batch functions
                         (job, mpijob, ..) the handler can also be specified in the `.run()` command, when not specified
                         the entire file will be executed (as main).
                         for nuclio functions the handler is in the form of module:function, defaults to 'main:handler'
    :param kind:         function runtime type string - nuclio, job, etc. (see docstring for all options)
    :param image:        base docker image to use for building the function container, defaults to None
    :param code_output:  specify '.' to generate python module from the current jupyter notebook
    :param embed_code:   indicates whether or not to inject the code directly into the function runtime spec,
                         defaults to True
    :param description:  short function description, defaults to ''
    :param requirements: list of python packages or pip requirements file path, defaults to None
    :param categories:   list of categories for mlrun Function Hub, defaults to None
    :param labels:       immutable name/value pairs to tag the function with useful metadata, defaults to None
    :param with_doc:     indicates whether to document the function parameters, defaults to True
    :param ignored_tags: notebook cells to ignore when converting notebooks to py code (separated by ';')

    :return:
        pre-configured function object from a mlrun runtime class

    example::

        import mlrun

        # create job function object from notebook code and add doc/metadata
        fn = mlrun.code_to_function("file_utils", kind="job",
                                    handler="open_archive", image="mlrun/mlrun",
                                    description = "this function opens a zip archive into a local/mounted folder",
                                    categories = ["fileutils"],
                                    labels = {"author": "me"})

    example::

        import mlrun
        from pathlib import Path

        # create file
        Path("mover.py").touch()

        # create nuclio function object from python module call mover.py
        fn = mlrun.code_to_function("nuclio-mover", kind="nuclio",
                                    filename="mover.py", image="python:3.7",
                                    description = "this function moves files from one system to another",
                                    requirements = ["pandas"],
                                    labels = {"author": "me"})

    """
    filebase, _ = path.splitext(path.basename(filename))
    ignored_tags = ignored_tags or mlconf.ignored_notebook_tags

    def add_name(origin, name=""):
        name = filename or (name + ".ipynb")
        if not origin:
            return name
        return f"{origin}:{name}"

    def update_common(fn, spec):
        fn.spec.image = image or get_in(spec, "spec.image", "")
        fn.spec.build.base_image = get_in(spec, "spec.build.baseImage")
        fn.spec.build.commands = get_in(spec, "spec.build.commands")
        fn.spec.build.secret = get_in(spec, "spec.build.secret")

        if requirements:
            fn.with_requirements(requirements)

        if embed_code:
            fn.spec.build.functionSourceCode = get_in(
                spec, "spec.build.functionSourceCode"
            )

        if fn.kind != "local":
            fn.spec.env = get_in(spec, "spec.env")
            for vol in get_in(spec, "spec.volumes", []):
                fn.spec.volumes.append(vol.get("volume"))
                fn.spec.volume_mounts.append(vol.get("volumeMount"))

        fn.spec.description = description
        fn.metadata.project = project or mlconf.default_project
        fn.metadata.tag = tag
        fn.metadata.categories = categories
        fn.metadata.labels = labels or fn.metadata.labels

    def resolve_nuclio_subkind(kind):
        is_nuclio = kind.startswith("nuclio")
        subkind = kind[kind.find(":") + 1 :] if is_nuclio and ":" in kind else None
        if kind == RuntimeKinds.serving:
            is_nuclio = True
            subkind = serving_subkind
        return is_nuclio, subkind

    if (
        not embed_code
        and not code_output
        and (not filename or filename.endswith(".ipynb"))
    ):
        raise ValueError(
            "a valid code file must be specified "
            "when not using the embed_code option"
        )

    is_nuclio, subkind = resolve_nuclio_subkind(kind)
    code_origin = add_name(add_code_metadata(filename), name)

    name, spec, code = nuclio.build_file(
        filename,
        name=name,
        handler=handler or "handler",
        kind=subkind,
        ignored_tags=ignored_tags,
    )
    spec_kind = get_in(spec, "kind", "")
    if not kind and spec_kind not in ["", "Function"]:
        kind = spec_kind.lower()

        # if its a nuclio subkind, redo nb parsing
        is_nuclio, subkind = resolve_nuclio_subkind(kind)
        if is_nuclio:
            name, spec, code = nuclio.build_file(
                filename,
                name=name,
                handler=handler or "handler",
                kind=subkind,
                ignored_tags=ignored_tags,
            )

    if code_output:
        if code_output == ".":
            code_output = name + ".py"
        if filename == "" or filename.endswith(".ipynb"):
            with open(code_output, "w") as fp:
                fp.write(code)
        else:
            raise ValueError("code_output option is only used with notebooks")

    if is_nuclio:
        if subkind == serving_subkind:
            r = ServingRuntime()
        else:
            r = RemoteRuntime()
            r.spec.function_kind = subkind
        # default_handler is only used in :mlrun subkind, determine the handler to invoke in function.run()
        r.spec.default_handler = handler if subkind == "mlrun" else ""
        r.spec.function_handler = (
            handler if handler and ":" in handler else get_in(spec, "spec.handler")
        )
        if not embed_code:
            r.spec.source = filename
        nuclio_runtime = get_in(spec, "spec.runtime")
        if nuclio_runtime and not nuclio_runtime.startswith("py"):
            r.spec.nuclio_runtime = nuclio_runtime
        if not name:
            raise ValueError("name must be specified")
        r.metadata.name = name
        r.spec.build.code_origin = code_origin
        r.spec.build.origin_filename = filename or (name + ".ipynb")
        update_common(r, spec)
        return r

    if kind is None or kind in ["", "Function"]:
        raise ValueError("please specify the function kind")
    elif kind in RuntimeKinds.all():
        r = get_runtime_class(kind)()
    else:
        raise ValueError(f"unsupported runtime ({kind})")

    name, spec, code = nuclio.build_file(filename, name=name, ignored_tags=ignored_tags)

    if not name:
        raise ValueError("name must be specified")
    h = get_in(spec, "spec.handler", "").split(":")
    r.handler = h[0] if len(h) <= 1 else h[1]
    r.metadata = get_in(spec, "spec.metadata")
    r.metadata.name = name
    build = r.spec.build
    build.code_origin = code_origin
    build.origin_filename = filename or (name + ".ipynb")
    build.extra = get_in(spec, "spec.build.extra")
    if not embed_code:
        if code_output:
            r.spec.command = code_output
        else:
            r.spec.command = filename

    build.image = get_in(spec, "spec.build.image")
    update_common(r, spec)
    r.verify_base_image()

    if with_doc:
        update_function_entry_points(r, code)
    r.spec.default_handler = handler
    return r


def run_pipeline(
    pipeline,
    arguments=None,
    project=None,
    experiment=None,
    run=None,
    namespace=None,
    artifact_path=None,
    ops=None,
    url=None,
    # TODO: deprecated, remove in 1.5.0
    ttl=None,
    remote: bool = True,
    cleanup_ttl=None,
):
    """remote KubeFlow pipeline execution

    Submit a workflow task to KFP via mlrun API service

    :param pipeline:   KFP pipeline function or path to .yaml/.zip pipeline file
    :param arguments:  pipeline arguments
    :param project:    name of project
    :param experiment: experiment name
    :param run:        optional, run name
    :param namespace:  Kubernetes namespace (if not using default)
    :param url:        optional, url to mlrun API service
    :param artifact_path:  target location/url for mlrun artifacts
    :param ops:        additional operators (.apply() to all pipeline functions)
    :param ttl:        pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                       workflow and all its resources are deleted) (deprecated, use cleanup_ttl instead)
    :param remote:     read kfp data from mlrun service (default=True)
    :param cleanup_ttl:
                       pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                       workflow and all its resources are deleted)

    :returns: kubeflow pipeline id
    """

    if ttl:
        warnings.warn(
            "'ttl' is deprecated, use 'cleanup_ttl' instead. "
            "This will be removed in 1.5.0",
            # TODO: Remove this in 1.5.0
            FutureWarning,
        )

    artifact_path = artifact_path or mlconf.artifact_path
    project = project or mlconf.default_project
    artifact_path = mlrun.utils.helpers.fill_artifact_path_template(
        artifact_path, project or mlconf.default_project
    )
    if artifact_path and "{{run.uid}}" in artifact_path:
        artifact_path.replace("{{run.uid}}", "{{workflow.uid}}")
    if not artifact_path:
        raise ValueError("artifact path was not specified")

    namespace = namespace or mlconf.namespace
    arguments = arguments or {}

    if remote or url:
        mldb = mlrun.db.get_run_db(url)
        if mldb.kind != "http":
            raise ValueError(
                "run pipeline require access to remote api-service"
                ", please set the dbpath url"
            )
        id = mldb.submit_pipeline(
            project,
            pipeline,
            arguments,
            experiment=experiment,
            run=run,
            namespace=namespace,
            ops=ops,
            artifact_path=artifact_path,
            cleanup_ttl=cleanup_ttl or ttl,
        )

    else:
        client = Client(namespace=namespace)
        if isinstance(pipeline, str):
            experiment = client.create_experiment(name=experiment)
            run_result = client.run_pipeline(
                experiment.id, run, pipeline, params=arguments
            )
        else:
            conf = new_pipe_metadata(
                artifact_path=artifact_path, cleanup_ttl=ttl, op_transformers=ops
            )
            run_result = client.create_run_from_pipeline_func(
                pipeline,
                arguments,
                run_name=run,
                experiment_name=experiment,
                pipeline_conf=conf,
            )

        id = run_result.run_id
    logger.info(f"Pipeline run id={id}, check UI for progress")
    return id


def wait_for_pipeline_completion(
    run_id,
    timeout=60 * 60,
    expected_statuses: List[str] = None,
    namespace=None,
    remote=True,
    project: str = None,
):
    """Wait for Pipeline status, timeout in sec

    :param run_id:     id of pipelines run
    :param timeout:    wait timeout in sec
    :param expected_statuses:  list of expected statuses, one of [ Succeeded | Failed | Skipped | Error ], by default
                               [ Succeeded ]
    :param namespace:  k8s namespace if not default
    :param remote:     read kfp data from mlrun service (default=True)
    :param project:    the project of the pipeline

    :return: kfp run dict
    """
    if expected_statuses is None:
        expected_statuses = [RunStatuses.succeeded]
    namespace = namespace or mlconf.namespace
    logger.debug(
        f"Waiting for run completion."
        f" run_id: {run_id},"
        f" project: {project},"
        f" expected_statuses: {expected_statuses},"
        f" timeout: {timeout},"
        f" remote: {remote},"
        f" namespace: {namespace}"
    )

    if remote:
        mldb = mlrun.db.get_run_db()

        def get_pipeline_if_completed(run_id, namespace=namespace):
            resp = mldb.get_pipeline(run_id, namespace=namespace, project=project)
            status = resp["run"]["status"]
            show_kfp_run(resp, clear_output=True)
            if status not in RunStatuses.stable_statuses():
                # TODO: think of nicer liveness indication and make it re-usable
                # log '.' each retry as a liveness indication
                logger.debug(".")
                raise RuntimeError("pipeline run has not completed yet")

            return resp

        if mldb.kind != "http":
            raise ValueError(
                "get pipeline require access to remote api-service"
                ", please set the dbpath url"
            )

        resp = retry_until_successful(
            10,
            timeout,
            logger,
            False,
            get_pipeline_if_completed,
            run_id,
            namespace=namespace,
        )
    else:
        client = Client(namespace=namespace)
        resp = client.wait_for_run_completion(run_id, timeout)
        if resp:
            resp = resp.to_dict()
            resp = format_summary_from_kfp_run(resp)
        show_kfp_run(resp)

    status = resp["run"]["status"] if resp else "unknown"
    message = resp["run"].get("message", "")
    if expected_statuses:
        if status not in expected_statuses:
            raise RuntimeError(
                f"Pipeline run status {status}{', ' + message if message else ''}"
            )

    logger.debug(
        f"Finished waiting for pipeline completion."
        f" run_id: {run_id},"
        f" status: {status},"
        f" message: {message},"
        f" namespace: {namespace}"
    )

    return resp


def get_pipeline(
    run_id,
    namespace=None,
    format_: Union[
        str, mlrun.api.schemas.PipelinesFormat
    ] = mlrun.api.schemas.PipelinesFormat.summary,
    project: str = None,
    remote: bool = True,
):
    """Get Pipeline status

    :param run_id:     id of pipelines run
    :param namespace:  k8s namespace if not default
    :param format_:    Format of the results. Possible values are:
            - ``summary`` (default value) - Return summary of the object data.
            - ``full`` - Return full pipeline object.
    :param project:    the project of the pipeline run
    :param remote:     read kfp data from mlrun service (default=True)

    :return: kfp run dict
    """
    namespace = namespace or mlconf.namespace
    if remote:
        mldb = mlrun.db.get_run_db()
        if mldb.kind != "http":
            raise ValueError(
                "get pipeline require access to remote api-service"
                ", please set the dbpath url"
            )

        resp = mldb.get_pipeline(
            run_id, namespace=namespace, format_=format_, project=project
        )

    else:
        client = Client(namespace=namespace)
        resp = client.get_run(run_id)
        if resp:
            resp = resp.to_dict()
            if (
                not format_
                or format_ == mlrun.api.schemas.PipelinesFormat.summary.value
            ):
                resp = format_summary_from_kfp_run(resp)

    show_kfp_run(resp)
    return resp


def list_pipelines(
    full=False,
    page_token="",
    page_size=None,
    sort_by="",
    filter_="",
    namespace=None,
    project="*",
    format_: mlrun.api.schemas.PipelinesFormat = mlrun.api.schemas.PipelinesFormat.metadata_only,
) -> Tuple[int, Optional[int], List[dict]]:
    """List pipelines

    :param full:       Deprecated, use `format_` instead. if True will set `format_` to full, otherwise `format_` will
                       be used
    :param page_token: A page token to request the next page of results. The token is acquired from the nextPageToken
                       field of the response from the previous call or can be omitted when fetching the first page.
    :param page_size:  The number of pipelines to be listed per page. If there are more pipelines than this number, the
                       response message will contain a nextPageToken field you can use to fetch the next page.
    :param sort_by:    Can be format of "field_name", "field_name asc" or "field_name desc" (Example, "name asc"
                       or "id desc"). Ascending by default.
    :param filter_:    A url-encoded, JSON-serialized Filter protocol buffer, see:
                       [filter.proto](https://github.com/kubeflow/pipelines/ blob/master/backend/api/filter.proto).
    :param namespace:  Kubernetes namespace if other than default
    :param project:    Can be used to retrieve only specific project pipelines. "*" for all projects. Note that
                       filtering by project can't be used together with pagination, sorting, or custom filter.
    :param format_:    Control what will be returned (full/metadata_only/name_only)
    """
    if full:
        format_ = mlrun.api.schemas.PipelinesFormat.full
    run_db = mlrun.db.get_run_db()
    pipelines = run_db.list_pipelines(
        project, namespace, sort_by, page_token, filter_, format_, page_size
    )
    return pipelines.total_size, pipelines.next_page_token, pipelines.runs


def get_object(url, secrets=None, size=None, offset=0, db=None):
    """get mlrun dataitem body (from path/url)"""
    stores = store_manager.set(secrets, db=db)
    return stores.object(url=url).get(size, offset)


def get_dataitem(url, secrets=None, db=None) -> mlrun.datastore.DataItem:
    """get mlrun dataitem object (from path/url)"""
    stores = store_manager.set(secrets, db=db)
    return stores.object(url=url)


def download_object(url, target, secrets=None):
    """download mlrun dataitem (from path/url to target path)"""
    stores = store_manager.set(secrets)
    stores.object(url=url).download(target_path=target)


def wait_for_runs_completion(runs: list, sleep=3, timeout=0, silent=False):
    """wait for multiple runs to complete

    Note: need to use `watch=False` in `.run()` so the run will not wait for completion

    example::

        # run two training functions in parallel and wait for the results
        inputs = {'dataset': cleaned_data}
        run1 = train.run(name='train_lr', inputs=inputs, watch=False,
                         params={'model_pkg_class': 'sklearn.linear_model.LogisticRegression',
                                 'label_column': 'label'})
        run2 = train.run(name='train_lr', inputs=inputs, watch=False,
                         params={'model_pkg_class': 'sklearn.ensemble.RandomForestClassifier',
                                 'label_column': 'label'})
        completed = wait_for_runs_completion([run1, run2])

    :param runs:    list of run objects (the returned values of function.run())
    :param sleep:   time to sleep between checks (in seconds)
    :param timeout: maximum time to wait in seconds (0 for unlimited)
    :param silent:  set to True for silent exit on timeout
    :return: list of completed runs
    """
    completed = []
    total_time = 0
    while True:
        running = []
        for run in runs:
            state = run.state()
            if state in mlrun.runtimes.constants.RunStates.terminal_states():
                completed.append(run)
            else:
                running.append(run)
        if len(running) == 0:
            break
        time.sleep(sleep)
        total_time += sleep
        if timeout and total_time > timeout:
            if silent:
                break
            raise mlrun.errors.MLRunTimeoutError(
                "some runs did not reach terminal state on time"
            )
        runs = running

    return completed


class ArtifactType(Enum):
    """
    Possible artifact types to log using the MLRun `context` decorator.
    """

    # Types:
    DATASET = "dataset"
    DIRECTORY = "directory"
    FILE = "file"
    OBJECT = "object"
    PLOT = "plot"
    RESULT = "result"

    # Constants:
    DEFAULT = RESULT


# Instruction types:
LogInstructionType = Union[
    Tuple[str, ArtifactType],
    Tuple[str, str],
    Tuple[str, ArtifactType, Dict[str, Any]],
    Tuple[str, str, Dict[str, Any]],
    str,
    None,
]
ParseInstructionType = Dict[str, Type]


class InputsParser:
    """
    A static class to hold all the common parsing functions - functions for parsing MLRun DataItem to the user desired
    type.
    """

    @staticmethod
    def parse_pandas_dataframe(data_item: DataItem) -> pd.DataFrame:
        """
        Parse an MLRun `DataItem` to a `pandas.DataFrame`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `pandas.DataFrame`.
        """
        return data_item.as_df()

    @staticmethod
    def parse_numpy_array(data_item: DataItem) -> np.ndarray:
        """
        Parse an MLRun `DataItem` to a `numpy.ndarray`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `numpy.ndarray`.
        """
        return data_item.as_df().to_numpy()

    @staticmethod
    def parse_dict(data_item: DataItem) -> dict:
        """
        Parse an MLRun `DataItem` to a `dict`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `dict`.
        """
        return data_item.as_df().to_dict()

    @staticmethod
    def parse_list(data_item: DataItem) -> list:
        """
        Parse an MLRun `DataItem` to a `list`.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as a `list`.
        """
        return data_item.as_df().to_numpy().tolist()

    @staticmethod
    def parse_object(data_item: DataItem) -> object:
        """
        Parse an MLRun `DataItem` to its unpickled object. The pickle file will be downloaded to a local temp
        directory and then loaded.

        :param data_item: The `DataItem` to parse.

        :returns: The `DataItem` as the original object that was pickled once it was logged.
        """
        object_file = data_item.local()
        with open(object_file, "rb") as pickle_file:
            obj = cloudpickle.load(pickle_file)
        return obj


class OutputsLogger:
    """
    A static class to hold all the common logging functions - functions for logging different objects by artifact type
    to MLRun.
    """

    @staticmethod
    def log_dataset(
        ctx: MLClientCtx,
        obj: Union[pd.DataFrame, np.ndarray, pd.Series, dict, list],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log an object as a dataset. The dataset wil lbe cast to a `pandas.DataFrame`. Supporting casting from
        `pandas.Series`, `numpy.ndarray`, `dict` and `list`.

        :param ctx:            The MLRun context to log with.
        :param obj:            The data to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_dataset`

        :raise MLRunInvalidArgumentError: If the type is not supported for being cast to `pandas.DataFrame`.
        """
        # Check for the object type:
        if not isinstance(obj, pd.DataFrame):
            if isinstance(obj, (np.ndarray, pd.Series, dict, list)):
                obj = pd.DataFrame(obj)
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The value requested to be logged as a dataset artifact is of type '{type(obj)}' and it "
                    f"cannot be logged as a dataset. Please parse it in your code into one `numpy.ndarray`, "
                    f"`pandas.DataFrame`, `pandas.Series`, `dict`, `list` before returning it so we can log it."
                )

        # Log the DataFrame object as a dataset:
        ctx.log_dataset(**logging_kwargs, key=key, df=obj)

    @staticmethod
    def log_directory(
        ctx: MLClientCtx,
        obj: Union[str, Path],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log a directory as a zip file. The zip file will be created at the current working directory. Once logged,
        it will be deleted.

        :param ctx:            The MLRun context to log with.
        :param obj:            The directory to zip path.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact` method.

        :raises MLRunInvalidArgumentError: In case the given path is not of a directory or do not exist.
        """
        # In case it is a `pathlib` path, parse to str:
        obj = str(obj)

        # Verify the path is of an existing directory:
        if not os.path.isdir(obj):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given path is not a directory: '{obj}'"
            )
        if not os.path.exists(obj):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given directory path do not exist: '{obj}'"
            )

        # Zip the directory:
        directory_zip_path = shutil.make_archive(
            base_name=key,
            format="zip",
            root_dir=os.path.abspath(obj),
        )

        # Log the zip file:
        ctx.log_artifact(**logging_kwargs, item=key, local_path=directory_zip_path)

        # Delete the zip file:
        os.remove(directory_zip_path)

    @staticmethod
    def log_file(
        ctx: MLClientCtx,
        obj: Union[str, Path],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log a file to MLRun.

        :param ctx:            The MLRun context to log with.
        :param obj:            The path of the file to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact` method.

        :raises MLRunInvalidArgumentError: In case the given path is not of a file or do not exist.
        """
        # In case it is a `pathlib` path, parse to str:
        obj = str(obj)

        # Verify the path is of an existing directory:
        if not os.path.isfile(obj):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given path is not a file: '{obj}'"
            )
        if not os.path.exists(obj):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given directory path do not exist: '{obj}'"
            )

        # Log the zip file:
        ctx.log_artifact(**logging_kwargs, item=key, local_path=os.path.abspath(obj))

    @staticmethod
    def log_object(ctx: MLClientCtx, obj, key: str, logging_kwargs: dict):
        """
        Log an object as a pickle.

        :param ctx:            The MLRun context to log with.
        :param obj:            The object to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact` method.
        """
        ctx.log_artifact(
            **logging_kwargs,
            item=key,
            body=obj if isinstance(obj, (bytes, bytearray)) else cloudpickle.dumps(obj),
            format="pkl",
        )

    @staticmethod
    def log_plot(ctx: MLClientCtx, obj, key: str, logging_kwargs: dict):
        """
        Log an object as a plot. Currently, supporting plots produced by one the following modules: `matplotlib`,
        `seaborn`, `plotly` and `bokeh`.

        :param ctx:            The MLRun context to log with.
        :param obj:            The plot to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_artifact`.

        :raise MLRunInvalidArgumentError: If the object type is not supported (meaning the plot was not produced by
                                          one of the supported modules).
        """
        # Create the plot artifact according to the module produced the object:
        artifact = None

        # `matplotlib` and `seaborn`:
        try:
            import matplotlib.pyplot as plt

            from mlrun.artifacts import PlotArtifact

            # Get the figure:
            figure = None
            if isinstance(obj, plt.Figure):
                figure = obj
            elif isinstance(obj, plt.Axes):
                if hasattr(obj, "get_figure"):
                    figure = obj.get_figure()
                elif hasattr(obj, "figure"):
                    figure = obj.figure
                elif hasattr(obj, "fig"):
                    figure = obj.fig

            # Create the artifact:
            if figure is not None:
                artifact = PlotArtifact(key=key, body=figure)
        except ModuleNotFoundError:
            pass

        # `plotly`:
        if artifact is None:
            try:
                import plotly

                from mlrun.artifacts import PlotlyArtifact

                if isinstance(obj, plotly.graph_objs.Figure):
                    artifact = PlotlyArtifact(key=key, figure=obj)
            except ModuleNotFoundError:
                pass

        # `bokeh`:
        if artifact is None:
            try:
                import bokeh.plotting as bokeh_plt

                from mlrun.artifacts import BokehArtifact

                if isinstance(obj, bokeh_plt.Figure):
                    artifact = BokehArtifact(key=key, figure=obj)
            except ModuleNotFoundError:
                pass
            except ImportError:
                logger.warn(
                    "Bokeh installation is ignored. If needed, "
                    "make sure you have the required version with `pip install mlrun[bokeh]`"
                )

        # Log the artifact:
        if artifact is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given plot is of type `{type(obj)}`. We currently support logging plots produced by one of "
                f"the following modules: `matplotlib`, `seaborn`, `plotly` and `bokeh`. You may try to save the "
                f"plot to file and log it as a file instead."
            )
        ctx.log_artifact(**logging_kwargs, item=artifact)

    @staticmethod
    def log_result(
        ctx: MLClientCtx,
        obj: Union[int, float, str, list, tuple, dict, np.ndarray],
        key: str,
        logging_kwargs: dict,
    ):
        """
        Log an object as a result. The objects value will be cast to a serializable version of itself. Supporting:
        int, float, str, list, tuple, dict, numpy.ndarray

        :param ctx:            The MLRun context to log with.
        :param obj:            The value to log.
        :param key:            The key of the artifact.
        :param logging_kwargs: Additional keyword arguments to pass to the `context.log_result` method.
        """
        ctx.log_result(**logging_kwargs, key=key, value=obj)


class ContextHandler:
    """
    Private class for handling an MLRun context of a function that is wrapped in MLRun's `handler` decorator.

    The context handler have 3 duties:
      1. Check if the user used MLRun to run the wrapped function and if so, get the MLRun context.
      2. Parse the user's inputs (MLRun `DataItem`) to the function.
      3. Log the function's outputs to MLRun.

    The context handler use dictionaries to map objects to their logging / parsing function. The maps can be edited
    using the relevant `update_X` class method. If needed to add additional artifacts types, the `ArtifactType` class
    can be inherited and replaced as well using the `update_artifact_type_class` class method.
    """

    # The artifact type enum class to use:
    _ARTIFACT_TYPE_CLASS = ArtifactType
    # The map to use to get default artifact types of objects:
    _DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP = None
    # The map to use for logging an object by its type:
    _OUTPUTS_LOGGING_MAP = None
    # The map to use for parsing an object by its type:
    _INPUTS_PARSING_MAP = None

    @classmethod
    def update_artifact_type_class(cls, artifact_type_class: Type[ArtifactType]):
        """
        Update the artifact type enum class that the handler will use to specify new artifact types to log and parse.

        :param artifact_type_class: An enum inheriting from the `ArtifactType` enum.
        """
        cls._ARTIFACT_TYPE_CLASS = artifact_type_class

    @classmethod
    def update_default_objects_artifact_types_map(
        cls, updates: Dict[type, ArtifactType]
    ):
        """
        Enrich the default objects artifact types map with new objects types to support.

        :param updates: New objects types to artifact types to support.
        """
        if cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP is None:
            cls._init_default_objects_artifact_types_map()
        cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP.update(updates)

    @classmethod
    def update_outputs_logging_map(
        cls,
        updates: Dict[ArtifactType, Callable[[MLClientCtx, Any, str, dict], None]],
    ):
        """
        Enrich the outputs logging map with new artifact types to support. The outputs logging map is a dictionary of
        artifact type enum as key, and a function that will handle the given output. The function must accept 4 keyword
        arguments

        * ctx: `mlrun.MLClientCtx` - The MLRun context to log with.
        * obj: `Any` - The value / object to log.
        * key: `str` - The key of the artifact.
        * logging_kwargs: `dict` - Keyword arguments the user can pass in the instructions tuple.

        :param updates: New artifact types to support - a dictionary of artifact type enum as key, and a function that
                        will handle the given output to update the current map.
        """
        if cls._OUTPUTS_LOGGING_MAP is None:
            cls._init_outputs_logging_map()
        cls._OUTPUTS_LOGGING_MAP.update(updates)

    @classmethod
    def update_inputs_parsing_map(cls, updates: Dict[type, Callable[[DataItem], Any]]):
        """
        Enrich the inputs parsing map with new objects to support. The inputs parsing map is a dictionary of object
        types as key, and a function that will handle the given input. The function must accept 1 keyword argument
        (data_item: `mlrun.DataItem`) and return the relevant parsed object.

        :param updates: New object types to support - a dictionary of artifact type enum as key, and a function that
                        will handle the given input to update the current map.
        """
        if cls._INPUTS_PARSING_MAP is None:
            cls._init_inputs_parsing_map()
        cls._INPUTS_PARSING_MAP.update(updates)

    def __init__(self):
        """
        Initialize a context handler.
        """
        # Initialize the maps:
        if self._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP is None:
            self._init_default_objects_artifact_types_map()
        if self._OUTPUTS_LOGGING_MAP is None:
            self._init_outputs_logging_map()
        if self._INPUTS_PARSING_MAP is None:
            self._init_inputs_parsing_map()

        # Set up a variable to hold the context:
        self._context: MLClientCtx = None

    def look_for_context(self, args: tuple, kwargs: dict):
        """
        Look for an MLRun context (`mlrun.MLClientCtx`). The handler will look for a context in the given order:
          1. The given arguments.
          2. The given keyword arguments.
          3. If an MLRun RunTime was used the context will be located via the `mlrun.get_or_create_ctx` method.

        :param args:   The arguments tuple passed to the function.
        :param kwargs: The keyword arguments dictionary passed to the function.
        """
        # Search in the given arguments:
        for argument in args:
            if isinstance(argument, MLClientCtx):
                self._context = argument
                return

        # Search in the given keyword arguments:
        for argument_name, argument_value in kwargs.items():
            if isinstance(argument_value, MLClientCtx):
                self._context = argument_value
                return

        # Search if the function was triggered from an MLRun RunTime object by looking at the call stack:
        # Index 0: the current frame.
        # Index 1: the decorator's frame.
        # Index 2-...: If it is from mlrun.runtimes we can be sure it ran via MLRun, otherwise not.
        for callstack_frame in inspect.getouterframes(inspect.currentframe()):
            if os.path.join("mlrun", "runtimes", "") in callstack_frame.filename:
                self._context = mlrun.get_or_create_ctx("context")
                break

    def is_context_available(self) -> bool:
        """
        Check if a context was found by the method `look_for_context`.

        :returns: True if a context was found and False otherwise.
        """
        return self._context is not None

    def parse_inputs(
        self, args: tuple, kwargs: dict, expected_arguments_types: OrderedDict
    ) -> tuple:
        """
        Parse the given arguments and keyword arguments data items to the expected types.

        :param args:                     The arguments tuple passed to the function.
        :param kwargs:                   The keyword arguments dictionary passed to the function.
        :param expected_arguments_types: An ordered dictionary of the expected types of arguments.

        :returns: The parsed args (kwargs are parsed inplace).
        """
        # Parse the arguments:
        parsed_args = []
        expected_arguments_keys = list(expected_arguments_types.keys())
        for i, argument in enumerate(args):
            if (
                isinstance(argument, mlrun.DataItem)
                and expected_arguments_types[expected_arguments_keys[i]]
                != inspect._empty
            ):
                parsed_args.append(
                    self._parse_input(
                        data_item=argument,
                        expected_type=expected_arguments_types[
                            expected_arguments_keys[i]
                        ],
                    )
                )
                continue
            parsed_args.append(argument)
        parsed_args = tuple(parsed_args)  # `args` is expected to be a tuple.

        # Parse the keyword arguments:
        for key in kwargs.keys():
            if isinstance(kwargs[key], mlrun.DataItem) and expected_arguments_types[
                key
            ] not in [
                inspect._empty,
                mlrun.DataItem,
            ]:
                kwargs[key] = self._parse_input(
                    data_item=kwargs[key], expected_type=expected_arguments_types[key]
                )

        return parsed_args

    def log_outputs(
        self,
        outputs: list,
        logging_instructions: List[LogInstructionType],
    ):
        """
        Log the given outputs as artifacts with the stored context.

        :param outputs:              List of outputs to log.
        :param logging_instructions: List of logging instructions to use.
        """
        for obj, instructions in zip(outputs, logging_instructions):
            # Check if needed to log (not None):
            if instructions is None:
                continue
            # Parse the instructions:
            artifact_type = self._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP.get(
                type(obj), self._ARTIFACT_TYPE_CLASS.DEFAULT
            ).value
            key = None
            logging_kwargs = {}
            if isinstance(instructions, str):
                # A string with a template of "{key}" or "{key}: {artifact_type}":
                if ":" in instructions:
                    key, artifact_type = instructions.split(":", 1)
                    # Remove spaces after ':':
                    artifact_type = artifact_type.lstrip(" ")
                else:
                    key = instructions
            elif isinstance(instructions, tuple):
                # A tuple of [0] - key, [1] - artifact type, [2] - context log kwargs:
                key = instructions[0]
                artifact_type = instructions[1]
                if len(instructions) > 2:
                    logging_kwargs = instructions[2]
            # Check if the object to log is None (None values are only logged if the artifact type is Result):
            if obj is None and artifact_type != ArtifactType.RESULT.value:
                continue
            # Log:
            self._log_output(
                obj=obj,
                artifact_type=artifact_type,
                key=key,
                logging_kwargs=logging_kwargs,
            )

    def set_labels(self, labels: Dict[str, str]):
        """
        Set the given labels with the stored context.

        :param labels: The labels to set.
        """
        for key, value in labels.items():
            self._context.set_label(key=key, value=value)

    @classmethod
    def _init_default_objects_artifact_types_map(cls):
        """
        Initialize the default objects artifact types map with the basic classes supported by MLRun. In addition, it
        will try to support further common packages that are not required in MLRun.
        """
        # Initialize the map with the default classes:
        cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP = {
            pd.DataFrame: ArtifactType.DATASET,
            pd.Series: ArtifactType.DATASET,
            np.ndarray: ArtifactType.DATASET,
            dict: ArtifactType.RESULT,
            list: ArtifactType.RESULT,
            tuple: ArtifactType.RESULT,
            str: ArtifactType.RESULT,
            int: ArtifactType.RESULT,
            float: ArtifactType.RESULT,
            bytes: ArtifactType.OBJECT,
            bytearray: ArtifactType.OBJECT,
        }

        # Try to enrich it with further classes according ot the user's environment:
        try:
            import matplotlib.pyplot as plt

            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[plt.Figure] = ArtifactType.PLOT
            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[plt.Axes] = ArtifactType.PLOT
        except ModuleNotFoundError:
            pass
        try:
            import plotly

            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[
                plotly.graph_objs.Figure
            ] = ArtifactType.PLOT
        except ModuleNotFoundError:
            pass
        try:
            import bokeh.plotting as bokeh_plt

            cls._DEFAULT_OBJECTS_ARTIFACT_TYPES_MAP[
                bokeh_plt.Figure
            ] = ArtifactType.PLOT
        except ModuleNotFoundError:
            pass
        except ImportError:
            logger.warn(
                "Bokeh installation is ignored. If needed, "
                "make sure you have the required version with `pip install mlrun[bokeh]`"
            )

    @classmethod
    def _init_outputs_logging_map(cls):
        """
        Initialize the outputs logging map for the basic artifact types supported by MLRun.
        """
        cls._OUTPUTS_LOGGING_MAP = {
            ArtifactType.DATASET: OutputsLogger.log_dataset,
            ArtifactType.DIRECTORY: OutputsLogger.log_directory,
            ArtifactType.FILE: OutputsLogger.log_file,
            ArtifactType.OBJECT: OutputsLogger.log_object,
            ArtifactType.PLOT: OutputsLogger.log_plot,
            ArtifactType.RESULT: OutputsLogger.log_result,
        }

    @classmethod
    def _init_inputs_parsing_map(cls):
        """
        Initialize the inputs parsing map with the basic classes supported by MLRun.
        """
        cls._INPUTS_PARSING_MAP = {
            pd.DataFrame: InputsParser.parse_pandas_dataframe,
            np.ndarray: InputsParser.parse_numpy_array,
            dict: InputsParser.parse_dict,
            list: InputsParser.parse_list,
            object: InputsParser.parse_object,
        }

    def _parse_input(self, data_item: DataItem, expected_type: type) -> Any:
        """
        Parse the given data frame to the expected type. By default, it will be parsed to an object (will be treated as
        a pickle).

        :param data_item:     The data item to parse.
        :param expected_type: THe expected type to parse to.

        :returns: The parsed data item.

        :raises MLRunRuntimeError: If an error was raised during the parsing function.
        """
        try:
            return self._INPUTS_PARSING_MAP.get(
                expected_type, self._INPUTS_PARSING_MAP[object]
            )(data_item=data_item)
        except Exception as exception:
            raise mlrun.errors.MLRunRuntimeError(
                f"MLRun tried to parse a `DataItem` of type '{expected_type}' but failed. Be sure the item was "
                f"logged correctly - as the type you are trying to parse it back to. In general, python objects should "
                f"be logged as pickles."
            ) from exception

    def _log_output(
        self,
        obj,
        artifact_type: Union[ArtifactType, str],
        key: str,
        logging_kwargs: Dict[str, Any],
    ):
        """
        Log the given object to MLRun as the given artifact type with the provided key. The key can be part of a
        logging keyword arguments to pass to the relevant context logging function.

        :param obj:           The object to log.
        :param artifact_type: The artifact type to log the object as.
        :param key:           The key (name) of the artifact or a logging kwargs to use when logging the artifact.

        :raises MLRunInvalidArgumentError: If a key was provided in the logging kwargs.
        :raises MLRunRuntimeError:         If an error was raised during the logging function.
        """
        # Get the artifact type (will also verify the artifact type is valid):
        artifact_type = self._ARTIFACT_TYPE_CLASS(artifact_type)

        # Check if 'key' or 'item' were given the logging kwargs:
        if "key" in logging_kwargs or "item" in logging_kwargs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "When passing logging keyword arguments, both 'key' and 'item' (according to the context method) "
                "cannot be added to the dictionary as the key is given on its own."
            )

        # Use the logging map to log the object:
        try:
            self._OUTPUTS_LOGGING_MAP[artifact_type](
                ctx=self._context,
                obj=obj,
                key=key,
                logging_kwargs=logging_kwargs,
            )
        except Exception as exception:
            raise mlrun.errors.MLRunRuntimeError(
                f"MLRun tried to log '{key}' as '{artifact_type.value}' but failed. If you didn't provide the artifact "
                f"type and the default one does not fit, try to select the correct type from the enum `ArtifactType`."
            ) from exception


def handler(
    labels: Dict[str, str] = None,
    outputs: List[LogInstructionType] = None,
    inputs: Union[bool, ParseInstructionType] = True,
):
    """
    MLRun's handler is a decorator to wrap a function and enable setting labels, automatic `mlrun.DataItem` parsing and
    outputs logging.

    :param labels:  Labels to add to the run. Expecting a dictionary with the labels names as keys. Default: None.
    :param outputs: Logging configurations for the function's returned values. Expecting a list of tuples and None
                    values:

                    * str - A string in the format of '{key}:{artifact_type}'. If a string was given without ':' it will
                            indicate the key and the artifact type will be according to the returned value
                            type.
                    * tuple - A tuple of:

                      * [0]: str - The key (name) of the artifact to use for the logged output.
                      * [1]: Union[`ArtifactType`, str] = "result" - An `ArtifactType` enum or an equivalent
                        string, that indicates how to log the returned value. The artifact types can be one of:

                        * DATASET = "dataset"
                        * DIRECTORY = "directory"
                        * FILE = "file"
                        * OBJECT = "object"
                        * PLOT = "plot"
                        * RESULT = "result".

                      * [2]: Optional[Dict[str, Any]] - A keyword arguments dictionary with the properties to pass to
                        the relevant logging function (one of `context.log_artifact`, `context.log_result`,
                        `context.log_dataset`).

                    * None - Do not log the output.

                    The list length must be equal to the total amount of returned values from the function. Default is
                    None - meaning no outputs will be logged.

    :param inputs: Parsing configurations for the arguments passed as inputs via the `run` method of an MLRun function.
                   Can be passed as a boolean value or a dictionary:

                   * True - Parse all found inputs to the assigned type hint in the function's signature. If there is no
                            type hint assigned, the value will remain an `mlrun.DataItem`.
                   * False - Do not parse inputs, leaving the inputs as `mlrun.DataItem`.
                   * Dict[str, Type] - A dictionary with argument name as key and the expected type to parse the
                                       `mlrun.DataItem` to.

                   Default: True.

    Example::

            import mlrun

            @mlrun.handler(outputs=["my_array", None, "my_multiplier"])
            def my_handler(array: np.ndarray, m: int):
                array = array * m
                m += 1
                return array, "I won't be logged", m

            >>> mlrun_function = mlrun.code_to_function("my_code.py", kind="job")
            >>> run_object = mlrun_function.run(
            ...     handler="my_handler",
            ...     inputs={"array": "store://my_array_Artifact"},
            ...     params={"m": 2}
            ... )
            >>> run_object.outputs
            {'my_multiplier': 3, 'my_array': 'store://...'}
    """

    def decorator(func: Callable):
        def wrapper(*args: tuple, **kwargs: dict):
            nonlocal labels
            nonlocal outputs
            nonlocal inputs

            # Set default `inputs` - inspect the full signature and add the user's input on top of it::
            func_signature = inspect.signature(func)
            if inputs:
                parameters = OrderedDict(
                    {
                        parameter.name: parameter.annotation
                        for parameter in func_signature.parameters.values()
                    }
                )
                if isinstance(inputs, dict):
                    parameters.update(**inputs)
                inputs = parameters

            # Create a context handler and look for a context:
            context_handler = ContextHandler()
            context_handler.look_for_context(args=args, kwargs=kwargs)

            # If an MLRun context is found, parse arguments pre-run (kwargs are parsed inplace):
            if context_handler.is_context_available() and inputs:
                args = context_handler.parse_inputs(
                    args=args, kwargs=kwargs, expected_arguments_types=inputs
                )

            # Call the original function and get the returning values:
            func_outputs = func(*args, **kwargs)

            # If an MLRun context is found, set the given labels and log the returning values to MLRun via the context:
            if context_handler.is_context_available():
                if labels:
                    context_handler.set_labels(labels=labels)
                if outputs:
                    context_handler.log_outputs(
                        outputs=func_outputs
                        if isinstance(func_outputs, tuple)
                        else [func_outputs],
                        logging_instructions=outputs,
                    )
                return  # Do not return any values as the function ran via MLRun.
            return func_outputs

        # Make sure to pass the wrapped function's signature (argument list, type hints and doc strings) to the wrapper:
        wrapper = functools.wraps(func)(wrapper)

        return wrapper

    return decorator
