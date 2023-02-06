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

import importlib.util as imputil
import inspect
import json
import os
import socket
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from copy import copy
from io import StringIO
from os import environ, remove
from pathlib import Path
from subprocess import PIPE, Popen
from sys import executable

from distributed import Client, as_completed
from nuclio import Event

import mlrun
from mlrun.lists import RunList

from ..errors import err_to_str
from ..execution import MLClientCtx
from ..model import RunObject
from ..utils import get_handler_extended, get_in, logger, set_paths
from ..utils.clones import extract_source
from .base import BaseRuntime, FunctionSpec, spec_fields
from .kubejob import KubejobRuntime
from .remotesparkjob import RemoteSparkRuntime
from .utils import RunError, global_context, log_std


class ParallelRunner:
    def _get_handler(self, handler, context):
        return handler

    def _get_dask_client(self, options):
        if options.dask_cluster_uri:
            function = mlrun.import_function(options.dask_cluster_uri)
            return function.client, function.metadata.name
        return Client(), None

    def _parallel_run_many(
        self, generator, execution: MLClientCtx, runobj: RunObject
    ) -> RunList:
        if self.spec.build.source and generator.options.dask_cluster_uri:
            # the attached dask cluster will not have the source code when we clone the git on run
            raise mlrun.errors.MLRunRuntimeError(
                "Cannot load source code into remote Dask at runtime use, "
                "function.deploy() to add the code into the image instead"
            )
        results = RunList()
        tasks = generator.generate(runobj)
        handler = runobj.spec.handler
        self._force_handler(handler)
        set_paths(self.spec.pythonpath)
        handler = self._get_handler(handler, execution)

        client, function_name = self._get_dask_client(generator.options)
        parallel_runs = generator.options.parallel_runs or 4
        queued_runs = 0
        num_errors = 0

        def process_result(future):
            nonlocal num_errors
            resp, sout, serr = future.result()
            runobj = RunObject.from_dict(resp)
            try:
                log_std(self._db_conn, runobj, sout, serr, skip=self.is_child)
                resp = self._update_run_state(resp)
            except RunError as err:
                resp = self._update_run_state(resp, err=err_to_str(err))
                num_errors += 1
            results.append(resp)
            if num_errors > generator.max_errors:
                logger.error("max errors reached, stopping iterations!")
                return True
            run_results = resp["status"].get("results", {})
            stop = generator.eval_stop_condition(run_results)
            if stop:
                logger.info(
                    f"reached early stop condition ({generator.options.stop_condition}), stopping iterations!"
                )
            return stop

        completed_iter = as_completed([])
        for task in tasks:
            task_struct = task.to_dict()
            project = get_in(task_struct, "metadata.project")
            uid = get_in(task_struct, "metadata.uid")
            iter = get_in(task_struct, "metadata.iteration", 0)
            mlrun.get_run_db().store_run(
                task_struct, uid=uid, project=project, iter=iter
            )
            resp = client.submit(
                remote_handler_wrapper, task.to_json(), handler, self.spec.workdir
            )
            completed_iter.add(resp)
            queued_runs += 1
            if queued_runs >= parallel_runs:
                future = next(completed_iter)
                early_stop = process_result(future)
                queued_runs -= 1
                if early_stop:
                    break

        for future in completed_iter:
            process_result(future)

        client.close()
        if function_name and generator.options.teardown_dask:
            logger.info("tearing down the dask cluster..")
            mlrun.get_run_db().delete_runtime_resources(
                kind="dask", object_id=function_name, force=True
            )

        return results


def remote_handler_wrapper(task, handler, workdir=None):
    if task and not isinstance(task, dict):
        task = json.loads(task)

    context = MLClientCtx.from_dict(
        task,
        autocommit=False,
        host=socket.gethostname(),
    )
    runobj = RunObject.from_dict(task)

    sout, serr = exec_from_params(handler, runobj, context, workdir)
    return context.to_dict(), sout, serr


class HandlerRuntime(BaseRuntime, ParallelRunner):
    kind = "handler"

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        handler = runobj.spec.handler
        self._force_handler(handler)
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        environ["MLRUN_META_TMPFILE"] = tmp
        set_paths(self.spec.pythonpath)

        context = MLClientCtx.from_dict(
            runobj.to_dict(),
            rundb=self.spec.rundb,
            autocommit=False,
            tmp=tmp,
            host=socket.gethostname(),
        )
        global_context.set(context)
        sout, serr = exec_from_params(handler, runobj, context, self.spec.workdir)
        log_std(self._db_conn, runobj, sout, serr, show=False)
        return context.to_dict()


class LocalFunctionSpec(FunctionSpec):
    _dict_fields = spec_fields + ["clone_target_dir"]

    def __init__(
        self,
        command=None,
        args=None,
        mode=None,
        default_handler=None,
        pythonpath=None,
        entry_points=None,
        description=None,
        workdir=None,
        build=None,
        clone_target_dir=None,
    ):
        super().__init__(
            command=command,
            args=args,
            mode=mode,
            build=build,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            default_handler=default_handler,
            pythonpath=pythonpath,
        )
        self.clone_target_dir = clone_target_dir


class LocalRuntime(BaseRuntime, ParallelRunner):
    kind = "local"
    _is_remote = False

    @property
    def spec(self) -> LocalFunctionSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", LocalFunctionSpec)

    def to_job(self, image=""):
        struct = self.to_dict()
        obj = KubejobRuntime.from_dict(struct)
        if image:
            obj.spec.image = image
        return obj

    def with_source_archive(self, source, workdir=None, handler=None, target_dir=None):
        """load the code from git/tar/zip archive at runtime or build

        :param source:     valid path to git, zip, or tar file, e.g.
                           git://github.com/mlrun/something.git
                           http://some/url/file.zip
        :param handler: default function handler
        :param workdir: working dir relative to the archive root or absolute (e.g. './subdir')
        :param target_dir: local target dir for repo clone (by default its <current-dir>/code)
        """
        self.spec.build.source = source
        self.spec.build.load_source_on_run = True
        if handler:
            self.spec.default_handler = handler
        if workdir:
            self.spec.workdir = workdir
        if target_dir:
            self.spec.clone_target_dir = target_dir

    def is_deployed(self):
        return True

    def _get_handler(self, handler, context):
        command = self.spec.command
        if not command and self.spec.build.functionSourceCode:
            # if the code is embedded in the function object extract or find it
            command, _ = mlrun.run.load_func_code(self)
        return load_module(command, handler, context)

    def _pre_run(self, runobj: RunObject, execution: MLClientCtx):
        workdir = self.spec.workdir
        execution._current_workdir = workdir
        execution._old_workdir = None

        if self.spec.build.source and not hasattr(self, "_is_run_local"):
            target_dir = extract_source(
                self.spec.build.source,
                self.spec.clone_target_dir,
                secrets=execution._secrets_manager,
            )
            if workdir and not workdir.startswith("/"):
                execution._current_workdir = os.path.join(target_dir, workdir)
            else:
                execution._current_workdir = workdir or target_dir

        if execution._current_workdir:
            execution._old_workdir = os.getcwd()
            workdir = os.path.realpath(execution._current_workdir)
            set_paths(workdir)
            os.chdir(workdir)
        else:
            set_paths(os.path.realpath("."))

        if (
            runobj.metadata.labels.get("kind") == RemoteSparkRuntime.kind
            and environ["MLRUN_SPARK_CLIENT_IGZ_SPARK"] == "true"
        ):
            from mlrun.runtimes.remotesparkjob import igz_spark_pre_hook

            igz_spark_pre_hook()

    def _post_run(self, results, execution: MLClientCtx):
        if execution._old_workdir:
            os.chdir(execution._old_workdir)

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        environ["MLRUN_EXEC_CONFIG"] = runobj.to_json()
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        environ["MLRUN_META_TMPFILE"] = tmp
        if self.spec.rundb:
            environ["MLRUN_DBPATH"] = self.spec.rundb

        handler = runobj.spec.handler
        handler_str = handler or "main"
        logger.debug(f"starting local run: {self.spec.command} # {handler_str}")
        pythonpath = self.spec.pythonpath

        if handler:
            set_paths(pythonpath)

            context = MLClientCtx.from_dict(
                runobj.to_dict(),
                rundb=self.spec.rundb,
                autocommit=False,
                tmp=tmp,
                host=socket.gethostname(),
            )
            try:
                fn = self._get_handler(handler, context)
                global_context.set(context)
                sout, serr = exec_from_params(fn, runobj, context)
                log_std(
                    self._db_conn, runobj, sout, serr, skip=self.is_child, show=False
                )
                return context.to_dict()
            # if RunError was raised it means that the error was raised as part of running the function
            # ( meaning the state was already updated to error ) therefore we just re-raise the error
            except RunError as err:
                raise err
            # this exception handling is for the case where we fail on pre-loading or post-running the function
            # and the state was not updated to error yet, therefore we update the state to error and raise as RunError
            except Exception as exc:
                # set_state here is mainly for sanity, as we will raise RunError which is expected to be handled
                # by the caller and will set the state to error ( in `update_run_state` )
                context.set_state(error=err_to_str(exc), commit=True)
                logger.error(f"run error, {traceback.format_exc()}")
                raise RunError(
                    "failed on pre-loading / post-running of the function"
                ) from exc

        else:
            command = self.spec.command
            command = command.format(**runobj.spec.parameters)
            logger.info(f"handler was not provided running main ({command})")
            arg_list = command.split()
            if self.spec.mode == "pass":
                cmd = arg_list
            else:
                cmd = [executable, "-u"] + arg_list

            env = None
            if pythonpath:
                if "PYTHONPATH" in environ:
                    pythonpath = f"{environ['PYTHONPATH']}:{pythonpath}"
                env = {"PYTHONPATH": pythonpath}
            if runobj.spec.verbose:
                if not env:
                    env = {}
                env["MLRUN_LOG_LEVEL"] = "DEBUG"

            args = self.spec.args
            if args:
                new_args = []
                for arg in args:
                    arg = arg.format(**runobj.spec.parameters)
                    new_args.append(arg)
                args = new_args

            sout, serr = run_exec(cmd, args, env=env, cwd=execution._current_workdir)
            log_std(self._db_conn, runobj, sout, serr, skip=self.is_child, show=False)

            try:
                with open(tmp) as fp:
                    resp = fp.read()
                remove(tmp)
                if resp:
                    return json.loads(resp)
                logger.error("empty context tmp file")
            except FileNotFoundError:
                logger.info("no context file found")
            return runobj.to_dict()


def load_module(file_name, handler, context):
    """Load module from file name"""
    module = None
    if file_name:
        path = Path(file_name)
        mod_name = path.name
        if path.suffix:
            mod_name = mod_name[: -len(path.suffix)]
        spec = imputil.spec_from_file_location(mod_name, file_name)
        if spec is None:
            raise RunError(f"cannot import from {file_name!r}")
        module = imputil.module_from_spec(spec)
        spec.loader.exec_module(module)

    class_args = {}
    if context:
        class_args = copy(context._parameters.get("_init_args", {}))

    return get_handler_extended(handler, context, class_args, namespaces=module)


def run_exec(cmd, args, env=None, cwd=None):
    if args:
        cmd += args
    out = ""
    if env and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    print("running:", cmd)
    process = Popen(cmd, stdout=PIPE, stderr=PIPE, env=os.environ, cwd=cwd)
    while True:
        nextline = process.stdout.readline()
        if not nextline and process.poll() is not None:
            break
        print(nextline.decode("utf-8"), end="")
        sys.stdout.flush()
        out += nextline.decode("utf-8")
    code = process.poll()

    err = process.stderr.read().decode("utf-8") if code != 0 else ""
    return out, err


class _DupStdout(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.buf = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buf.write(message)

    def flush(self):
        self.terminal.flush()


def exec_from_params(handler, runobj: RunObject, context: MLClientCtx, cwd=None):
    old_level = logger.level
    if runobj.spec.verbose:
        logger.set_logger_level("DEBUG")
    kwargs = get_func_arg(handler, runobj, context)

    stdout = _DupStdout()
    err = ""
    val = None
    old_dir = os.getcwd()
    with redirect_stdout(stdout):
        context.set_logger_stream(stdout)
        try:
            if cwd:
                os.chdir(cwd)
            val = handler(**kwargs)
            context.set_state("completed", commit=False)
        except Exception as exc:
            err = err_to_str(exc)
            logger.error(f"execution error, {traceback.format_exc()}")
            context.set_state(error=err, commit=False)
            logger.set_logger_level(old_level)

    stdout.flush()
    if cwd:
        os.chdir(old_dir)
    context.set_logger_stream(sys.stdout)
    if val:
        context.log_result("return", val)

    # completion will be ignored if error is set
    context.commit(completed=True)
    logger.set_logger_level(old_level)
    return stdout.buf.getvalue(), err


def get_func_arg(handler, runobj: RunObject, context: MLClientCtx, is_nuclio=False):
    params = runobj.spec.parameters or {}
    inputs = runobj.spec.inputs or {}
    kwargs = {}
    args = inspect.signature(handler).parameters

    for key in args.keys():
        if key == "context":
            kwargs[key] = context
        elif is_nuclio and key == "event":
            kwargs[key] = Event(runobj.to_dict())
        elif key in params:
            kwargs[key] = copy(params[key])
        elif key in inputs:
            obj = context.get_input(key, inputs[key])
            if type(args[key].default) is str or args[key].annotation == str:
                kwargs[key] = obj.local()
            else:
                kwargs[key] = context.get_input(key, inputs[key])
    return kwargs
