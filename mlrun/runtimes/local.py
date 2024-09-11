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
import inspect
import io
import json
import os
import socket
import sys
import tempfile
import threading
import traceback
from contextlib import redirect_stdout
from copy import copy
from io import StringIO
from os import environ, remove
from pathlib import Path
from subprocess import PIPE, Popen
from sys import executable

from nuclio import Event

import mlrun
import mlrun.common.constants as mlrun_constants
from mlrun.lists import RunList

from ..errors import err_to_str
from ..execution import MLClientCtx
from ..model import RunObject
from ..utils import get_handler_extended, get_in, logger, set_paths
from ..utils.clones import extract_source
from .base import BaseRuntime
from .kubejob import KubejobRuntime
from .remotesparkjob import RemoteSparkRuntime
from .utils import RunError, global_context, log_std


class ParallelRunner:
    def _get_trackers_manager(self):
        """
        useful to import and call get_trackers_manager from mlrun.track in order to avoid circular imports
        or imports in multiple spots mid-code
        :return: trackers_manager
        """
        from mlrun.track import TrackerManager

        return TrackerManager()

    def _get_handler(
        self, handler: str, context: MLClientCtx, embed_in_sys: bool = True
    ):
        return handler

    def _get_dask_client(self, options):
        from distributed import Client

        if options.dask_cluster_uri:
            function = mlrun.import_function(options.dask_cluster_uri)
            return function.client, function.metadata.name
        return Client(), None

    def _parallel_run_many(
        self, generator, execution: MLClientCtx, runobj: RunObject
    ) -> RunList:
        # TODO: this flow assumes we use dask - move it to dask runtime
        from distributed import as_completed

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
        handler = self._get_handler(handler, execution, embed_in_sys=False)

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
                logger.error("Max errors reached, stopping iterations!")
                return True
            run_results = resp["status"].get("results", {})
            stop = generator.eval_stop_condition(run_results)
            if stop:
                logger.info(
                    f"Reached early stop condition ({generator.options.stop_condition}), stopping iterations!"
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
            logger.info("Tearing down the dask cluster..")
            mlrun.get_run_db().delete_runtime_resources(
                project=self.metadata.project,
                kind=mlrun.runtimes.RuntimeKinds.dask,
                object_id=function_name,
                force=True,
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
        # Running tracking services pre run to detect if some of them should be used:
        trackers_manager = self._get_trackers_manager()
        trackers_manager.pre_run(context)
        sout, serr = exec_from_params(handler, runobj, context, self.spec.workdir)
        context = trackers_manager.post_run(context)
        log_std(self._db_conn, runobj, sout, serr, show=False)
        return context.to_dict()


class LocalRuntime(BaseRuntime, ParallelRunner):
    kind = "local"
    _is_remote = False

    def to_job(self, image=""):
        struct = self.to_dict()
        obj = KubejobRuntime.from_dict(struct)
        if image:
            obj.spec.image = image
        return obj

    def with_source_archive(self, source, workdir=None, handler=None, target_dir=None):
        """load the code from git/tar/zip archive at runtime or build

        :param source:      valid path to git, zip, or tar file, e.g.
                            git://github.com/mlrun/something.git
                            http://some/url/file.zip
        :param handler:     default function handler
        :param workdir:     working dir relative to the archive root (e.g. './subdir') or absolute
        :param target_dir:  local target dir for repo clone (by default its <current-dir>/code)
        """
        self.spec.build.source = source
        self.spec.build.load_source_on_run = True
        if handler:
            self.spec.default_handler = handler
        if workdir:
            self.spec.workdir = workdir
        if target_dir:
            self.spec.build.source_code_target_dir = target_dir

    def is_deployed(self):
        return True

    def _get_handler(
        self, handler: str, context: MLClientCtx, embed_in_sys: bool = True
    ):
        command = self.spec.command
        if not command and self.spec.build.functionSourceCode:
            # if the code is embedded in the function object extract or find it
            command, _ = mlrun.run.load_func_code(self)
        return load_module(command, handler, context, embed_in_sys=embed_in_sys)

    def _pre_run(self, runobj: RunObject, execution: MLClientCtx):
        workdir = self.spec.workdir
        execution._current_workdir = workdir
        execution._old_workdir = None

        # _is_run_local is set when the user specifies local=True in run()
        # in this case we don't want to extract the source code and contaminate the user's local dir
        if self.spec.build.source and not hasattr(self, "_is_run_local"):
            target_dir = extract_source(
                self.spec.build.source,
                self.spec.build.source_code_target_dir,
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
            runobj.metadata.labels.get(mlrun_constants.MLRunInternalLabels.kind)
            == RemoteSparkRuntime.kind
            and environ["MLRUN_SPARK_CLIENT_IGZ_SPARK"] == "true"
        ):
            from mlrun.runtimes.remotesparkjob import igz_spark_pre_hook

            igz_spark_pre_hook()

    def _post_run(self, results, execution: MLClientCtx):
        if execution._old_workdir:
            os.chdir(execution._old_workdir)

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        # we define a tmp file for mlrun to log its run, for easy access later
        environ["MLRUN_EXEC_CONFIG"] = runobj.to_json()
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        environ["MLRUN_META_TMPFILE"] = tmp
        if self.spec.rundb:
            environ["MLRUN_DBPATH"] = self.spec.rundb

        handler = runobj.spec.handler
        handler_str = handler or "main"
        logger.debug(f"Starting local run: {self.spec.command} # {handler_str}")
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
                # Running tracking services pre run to detect if some of them should be used:
                trackers_manager = self._get_trackers_manager()
                trackers_manager.pre_run(context)
                sout, serr = exec_from_params(fn, runobj, context)
                # If trackers where used, this is where we log all data collected to MLRun
                context = trackers_manager.post_run(context)
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
                logger.error(f"Run error, {traceback.format_exc()}")
                raise RunError(
                    "Failed on pre-loading / post-running of the function"
                ) from exc

        else:
            command = self.spec.command
            command = command.format(**runobj.spec.parameters)
            logger.info(f"Handler was not provided running main ({command})")
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
            # Running tracking services pre run to detect if some of them should be used:
            trackers_manager = self._get_trackers_manager()
            trackers_manager.pre_run(execution)
            sout, serr = run_exec(cmd, args, env=env, cwd=execution._current_workdir)

            run_obj_dict = runobj.to_dict()  # default value
            if os.path.isfile(tmp):
                with open(tmp) as fp:
                    resp = fp.read()
                remove(tmp)
                if resp:
                    run_obj_dict = json.loads(resp)
                else:
                    logger.debug("Empty context tmp file")
            else:
                logger.info("No context file found")

            # If trackers where used, this is where we log all data collected to MLRun
            run_obj_dict = trackers_manager.post_run(run_obj_dict)
            log_std(self._db_conn, runobj, sout, serr, skip=self.is_child, show=False)
            return run_obj_dict


def load_module(
    file_name: str,
    handler: str,
    context: MLClientCtx,
    embed_in_sys: bool = True,
):
    """
    Load module from filename
    :param file_name:       The module path to load
    :param handler:         The callable to load
    :param context:         Execution context
    :param embed_in_sys:    Embed the file-named module in sys.modules. This is not persistent with remote
                            environments and therefore can effect pickling.
    """
    module = None
    if file_name:
        path = Path(file_name)
        mod_name = path.name
        if path.suffix:
            mod_name = mod_name[: -len(path.suffix)]
        spec = imputil.spec_from_file_location(mod_name, file_name)
        if spec is None:
            raise RunError(f"Cannot import from {file_name!r}")
        module = imputil.module_from_spec(spec)
        if embed_in_sys:
            sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    class_args = {}
    if context:
        class_args = copy(context._parameters.get("_init_args", {}))

    return get_handler_extended(
        handler,
        context,
        class_args,
        namespaces=module,
        reload_modules=context._reset_on_run,
    )


def run_exec(cmd, args, env=None, cwd=None):
    if args:
        cmd += args
    if env and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    print("Running:", cmd)
    process = Popen(
        cmd, stdout=PIPE, stderr=PIPE, env=os.environ, cwd=cwd, universal_newlines=True
    )

    def read_stderr(stderr):
        while True:
            nextline = process.stderr.readline()
            if not nextline:
                break
            stderr.write(nextline)

    # ML-3710. We must read stderr in a separate thread to drain the stderr pipe so that the spawned process won't
    # hang if it tries to write more to stderr than the buffer size (default of approx 8kb).
    with io.StringIO() as stderr:
        stderr_consumer_thread = threading.Thread(target=read_stderr, args=[stderr])
        stderr_consumer_thread.start()

        with io.StringIO() as stdout:
            while True:
                nextline = process.stdout.readline()
                if not nextline:
                    break
                print(nextline, end="")
                sys.stdout.flush()
                stdout.write(nextline)
            out = stdout.getvalue()

        stderr_consumer_thread.join()
        err = stderr.getvalue()

    process.wait()

    # if we return anything for err, the caller will assume that the process failed
    code = process.poll()
    err = "" if code == 0 else err

    return out, err


class _DupStdout:
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
    try:
        if runobj.spec.verbose:
            logger.set_logger_level("DEBUG")

        # Prepare the inputs type hints (user may pass type hints as part of the input keys):
        runobj.spec.extract_type_hints_from_inputs()
        # Read the keyword arguments to pass to the function (combining params and inputs from the run spec):
        kwargs = get_func_arg(handler, runobj, context)

        stdout = _DupStdout()
        err = ""
        val = None
        old_dir = os.getcwd()
        commit = True
        with redirect_stdout(stdout):
            context.set_logger_stream(stdout)
            try:
                if cwd:
                    os.chdir(cwd)
                # Apply the MLRun handler decorator for parsing inputs using type hints and logging outputs using
                # log hints (Expected behavior: inputs are being parsed when they have type hints in code or given
                # by user. Outputs are logged only if log hints are provided by the user):
                if mlrun.mlconf.packagers.enabled:
                    val = mlrun.handler(
                        inputs=(
                            runobj.spec.inputs_type_hints
                            if runobj.spec.inputs_type_hints
                            else True  # True will use type hints if provided in user's code.
                        ),
                        outputs=(
                            runobj.spec.returns
                            if runobj.spec.returns
                            else None  # None will turn off outputs logging.
                        ),
                    )(handler)(**kwargs)
                else:
                    val = handler(**kwargs)
                context.set_state("completed", commit=False)
            except mlrun.errors.MLRunTaskCancelledError as exc:
                logger.warning("Run was aborted", err=err_to_str(exc))
                # Run was aborted, the state run state is updated by the abort job, no need to commit again
                context.set_state(
                    mlrun.common.runtimes.constants.RunStates.aborted, commit=False
                )
                commit = False
            except Exception as exc:
                err = err_to_str(exc)
                logger.error(f"Execution error, {traceback.format_exc()}")
                context.set_state(error=err, commit=False)

        stdout.flush()
        if cwd:
            os.chdir(old_dir)
        context.set_logger_stream(sys.stdout)
        if val:
            context.log_result("return", val)

        if commit:
            # completion will be ignored if error is set
            context.commit(completed=True)

    finally:
        logger.set_logger_level(old_level)
    return stdout.buf.getvalue(), err


def get_func_arg(handler, runobj: RunObject, context: MLClientCtx, is_nuclio=False):
    params = runobj.spec.parameters or {}
    inputs = runobj.spec.inputs or {}
    kwargs = {}
    args = inspect.signature(handler).parameters

    def _get_input_value(input_key: str):
        input_obj = context.get_input(input_key, inputs[input_key])
        # If there is no type hint annotation but there is a default value and its type is string, point the data
        # item to local downloaded file path (`local()` returns the downloaded temp path string):
        if args[input_key].annotation is inspect.Parameter.empty and isinstance(
            args[input_key].default, str
        ):
            return input_obj.local()
        else:
            return input_obj

    for key in args.keys():
        if key == "context":
            kwargs[key] = context
        elif is_nuclio and key == "event":
            kwargs[key] = Event(runobj.to_dict())
        elif key in params:
            kwargs[key] = copy(params[key])
        elif key in inputs:
            kwargs[key] = _get_input_value(key)

    list_of_params = list(args.values())
    if len(list_of_params) == 0:
        return kwargs

    # get the last parameter, as **kwargs can only be last in the function's parameters list
    last_param = list_of_params[-1]
    # VAR_KEYWORD meaning : A dict of keyword arguments that aren’t bound to any other parameter.
    # This corresponds to a **kwargs parameter in a Python function definition.
    if last_param.kind == last_param.VAR_KEYWORD:
        # if handler has **kwargs, pass all parameters provided by the user to the handler which were not already set
        # as part of the previous loop which handled all parameters which were explicitly defined in the handler
        for key in params:
            if key not in kwargs:
                kwargs[key] = copy(params[key])
        for key in inputs:
            if key not in kwargs:
                kwargs[key] = _get_input_value(key)
    return kwargs
