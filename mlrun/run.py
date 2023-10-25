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
import importlib.util as imputil
import json
import os
import pathlib
import socket
import tempfile
import time
import typing
import uuid
from base64 import b64decode
from copy import deepcopy
from os import environ, makedirs, path
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nuclio
import yaml
from deprecated import deprecated
from kfp import Client

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.helpers
from mlrun.kfpops import format_summary_from_kfp_run, show_kfp_run

from .common.helpers import parse_versioned_object_uri
from .config import config as mlconf
from .datastore import store_manager
from .errors import MLRunInvalidArgumentError, MLRunTimeoutError
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
    Spark3Runtime,
    get_runtime_class,
)
from .runtimes.databricks_job.databricks_runtime import DatabricksRuntime
from .runtimes.funcdoc import update_function_entry_points
from .runtimes.serving import serving_subkind
from .runtimes.utils import add_code_metadata, global_context
from .utils import (
    extend_hub_uri_if_needed,
    get_in,
    logger,
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


# TODO: remove in 1.6.0
@deprecated(
    version="1.4.0",
    reason="'run_local' will be removed in 1.6.0, use 'function.run(local=True)' instead",
    category=FutureWarning,
)
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
    notifications: List[mlrun.model.Notification] = None,
    returns: list = None,
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
    :param inputs:   Input objects to pass to the handler. Type hints can be given so the input will be parsed
                     during runtime from `mlrun.DataItem` to the given type hint. The type hint can be given
                     in the key field of the dictionary after a colon, e.g: "<key> : <type_hint>".
    :param artifact_path: default artifact output path
    :param mode:    Runtime mode for more details head to `mlrun.new_function`
    :param allow_empty_resources:   Allow passing non materialized set/vector as input to jobs
                                    (allows to have function which don't depend on having targets,
                                    e.g a function which accepts a feature vector uri and generate
                                     the offline vector e.g. parquet_ for it if it doesn't exist)
    :param notifications:   list of notifications to push when the run is completed
    :param returns:  List of configurations for how to log the returning values from the handler's run (as artifacts or
                     results). The list's length must be equal to the amount of returning objects. A configuration may
                     be given as:

                     * A string of the key to use to log the returning value as result or as an artifact. To specify
                       The artifact type, it is possible to pass a string in the following structure:
                       "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no artifact
                       type is specified, the object's default artifact type will be used.
                     * A dictionary of configurations to use when logging. Further info per object type and artifact
                       type can be given there. The artifact key must appear in the dictionary as "key": "the_key".

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
        returns=returns,
        artifact_path=artifact_path,
        notifications=notifications,
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

        fn = mlrun.import_function('hub://open-archive')
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
    labels: dict = None,
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
    :param labels:      dict of the context labels
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
            artifact_path = mlrun.utils.helpers.fill_project_path_template(
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
    labels = labels or {}
    for key, val in labels.items():
        ctx.set_label(key=key, value=val)
    global_context.set(ctx)
    return ctx


def import_function(url="", secrets=None, db="", project=None, new_name=None):
    """Create function object from DB or local/remote YAML file

    Functions can be imported from function repositories (mlrun Function Hub (formerly Marketplace) or local db),
    or be read from a remote URL (http(s), s3, git, v3io, ..) containing the function YAML

    special URLs::

        function hub:       hub://[{source}/]{name}[:{tag}]
        local mlrun db:     db://{project-name}/{name}[:{tag}]

    examples::

        function = mlrun.import_function("hub://auto-trainer")
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
    if url.startswith(mlrun.utils.db_prefix):
        url = url.removeprefix(mlrun.utils.db_prefix)
        _project, name, tag, hash_key = parse_versioned_object_uri(url)
        db = mlrun.db.get_run_db(db or mlrun.db.get_or_set_dburl(), secrets=secrets)
        runtime = db.get_function(name, _project, tag, hash_key)
        if not runtime:
            raise KeyError(f"function {name}:{tag} not found in the DB")
    else:
        url, is_hub_uri = extend_hub_uri_if_needed(url)
        runtime = import_function_to_dict(url, secrets)
    return compile_imported_function(project, runtime, new_name, from_hub=is_hub_uri)


def compile_imported_function(
    project_name, runtime, new_name: typing.Optional[str] = None, from_hub=False
) -> mlrun.runtimes.BaseRuntime:
    function = new_function(runtime=runtime)
    project_name = project_name or mlrun.mlconf.default_project

    # When we're importing from the hub we want to assign to a target project, otherwise any store on it will
    # simply default to the default project
    if from_hub:
        function.metadata.project = project_name
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
            if code_dir := path.dirname(code_file):
                makedirs(code_dir, exist_ok=True)
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
    requirements_file: str = "",
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
    :param mode:     runtime mode:
            * pass - will run the command as is in the container (not wrapped by mlrun), the command can use
                     params substitutions like {xparam} and will be replaced with the value of the xparam param
                     if a command is not specified, then image entrypoint shall be used.
    :param handler:  The default function handler to call for the job or nuclio function, in batch functions
                     (job, mpijob, ..) the handler can also be specified in the `.run()` command, when not specified
                     the entire file will be executed (as main).
                     for nuclio functions the handler is in the form of module:function, defaults to "main:handler"
    :param source:   valid absolute path or URL to git, zip, or tar file, e.g.
                     `git://github.com/mlrun/something.git`,
                     `http://some/url/file.zip`
                     note path source must exist on the image or exist locally when run is local
                     (it is recommended to use 'function.spec.workdir' when source is a filepath instead)
    :param requirements:        a list of python packages, defaults to None
    :param requirements_file:   path to a python requirements file
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
                f"Unsupported runtime ({kind}) or missing command, supported runtimes: {supported_runtimes}"
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
            raise MLRunInvalidArgumentError(
                "cannot set the handler for serving runtime"
            )
        elif kind in RuntimeKinds.nuclio_runtimes():
            runner.spec.function_handler = handler
        else:
            runner.spec.default_handler = handler

    if requirements or requirements_file:
        runner.with_requirements(
            requirements,
            requirements_file=requirements_file,
            prepare_image_for_deploy=False,
        )

    runner.prepare_image_for_deploy()
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
    requirements_file: str = "",
) -> Union[
    MpiRuntimeV1Alpha1,
    MpiRuntimeV1,
    RemoteRuntime,
    ServingRuntime,
    DaskCluster,
    KubejobRuntime,
    LocalRuntime,
    Spark3Runtime,
    RemoteSparkRuntime,
    DatabricksRuntime,
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

    Learn more about {Kinds of function (runtimes)](../concepts/functions-overview.html).

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
    :param requirements: a list of python packages
    :param requirements_file: path to a python requirements file
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
        fn.spec.filename = filename or get_in(spec, "spec.filename", "")
        fn.spec.build.base_image = get_in(spec, "spec.build.baseImage")
        fn.spec.build.commands = get_in(spec, "spec.build.commands")
        fn.spec.build.secret = get_in(spec, "spec.build.secret")

        if requirements or requirements_file:
            fn.with_requirements(requirements, requirements_file=requirements_file)

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

    if kind == RuntimeKinds.databricks and not embed_code:
        raise ValueError("databricks tasks only support embed_code=True")

    is_nuclio, subkind = resolve_nuclio_subkind(kind)
    code_origin = add_name(add_code_metadata(filename), name)

    name, spec, code = nuclio.build_file(
        filename,
        name=name,
        handler=handler or "handler",
        kind=subkind,
        ignored_tags=ignored_tags,
    )
    spec["spec"]["env"].append(
        {
            "name": "MLRUN_HTTPDB__NUCLIO__EXPLICIT_ACK",
            "value": mlrun.mlconf.httpdb.nuclio.explicit_ack,
        }
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
    build.extra_args = get_in(spec, "spec.build.extra_args")
    build.builder_env = get_in(spec, "spec.build.builder_env")
    if not embed_code:
        if code_output:
            r.spec.command = code_output
        else:
            r.spec.command = filename

    build.image = get_in(spec, "spec.build.image")
    update_common(r, spec)
    r.prepare_image_for_deploy()

    if with_doc:
        update_function_entry_points(r, code)
    r.spec.default_handler = handler
    return r


def _run_pipeline(
    pipeline,
    arguments=None,
    project=None,
    experiment=None,
    run=None,
    namespace=None,
    artifact_path=None,
    ops=None,
    url=None,
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
    :param cleanup_ttl:
                       pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                       workflow and all its resources are deleted)

    :returns: kubeflow pipeline id
    """
    mldb = mlrun.db.get_run_db(url)
    if mldb.kind != "http":
        raise ValueError(
            "run pipeline require access to remote api-service"
            ", please set the dbpath url"
        )

    pipeline_run_id = mldb.submit_pipeline(
        project,
        pipeline,
        arguments,
        experiment=experiment,
        run=run,
        namespace=namespace,
        ops=ops,
        artifact_path=artifact_path,
        cleanup_ttl=cleanup_ttl,
    )
    logger.info(f"Pipeline run id={pipeline_run_id}, check UI for progress")
    return pipeline_run_id


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

        def _wait_for_pipeline_completion():
            pipeline = mldb.get_pipeline(run_id, namespace=namespace, project=project)
            pipeline_status = pipeline["run"]["status"]
            show_kfp_run(pipeline, clear_output=True)
            if pipeline_status not in RunStatuses.stable_statuses():
                logger.debug(
                    "Waiting for pipeline completion",
                    run_id=run_id,
                    status=pipeline_status,
                )
                raise RuntimeError("pipeline run has not completed yet")

            return pipeline

        if mldb.kind != "http":
            raise ValueError(
                "get pipeline requires access to remote api-service"
                ", set the dbpath url"
            )

        resp = retry_until_successful(
            10,
            timeout,
            logger,
            False,
            _wait_for_pipeline_completion,
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
        str, mlrun.common.schemas.PipelinesFormat
    ] = mlrun.common.schemas.PipelinesFormat.summary,
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
                or format_ == mlrun.common.schemas.PipelinesFormat.summary.value
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
    format_: mlrun.common.schemas.PipelinesFormat = mlrun.common.schemas.PipelinesFormat.metadata_only,
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
        format_ = mlrun.common.schemas.PipelinesFormat.full
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


def wait_for_runs_completion(
    runs: typing.Union[list, typing.ValuesView], sleep=3, timeout=0, silent=False
):
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
            raise MLRunTimeoutError("some runs did not reach terminal state on time")
        runs = running

    return completed
