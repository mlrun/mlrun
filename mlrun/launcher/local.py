# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pathlib
from typing import Callable, Optional, Union

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.schedule
import mlrun.errors
import mlrun.launcher.client as launcher
import mlrun.run
import mlrun.runtimes.generators
import mlrun.utils.clones
import mlrun.utils.notifications
from mlrun.utils import logger


class ClientLocalLauncher(launcher.ClientBaseLauncher):
    """
    ClientLocalLauncher is a launcher that runs the job locally.
    Either on the user's machine (_is_run_local is True) or on a remote machine (_is_run_local is False).
    """

    def __init__(self, local: bool = False, **kwargs):
        """
        Initialize a ClientLocalLauncher.
        :param local:   True if the job runs on the user's local machine,
                        False if it runs on a remote machine (e.g. a dedicated k8s pod).
        """
        super().__init__(**kwargs)
        self._is_run_local = local

    def launch(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        task: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[Union[str, Callable]] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[dict[str, str]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        watch: Optional[bool] = True,
        schedule: Optional[
            Union[str, mlrun.common.schemas.schedule.ScheduleCronTrigger]
        ] = None,
        hyperparams: Optional[dict[str, list]] = None,
        hyper_param_options: Optional[mlrun.model.HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[dict[str, str]] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
        reset_on_run: Optional[bool] = None,
    ) -> "mlrun.run.RunObject":
        # do not allow local function to be scheduled
        if schedule is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unexpected {schedule=} parameter for local function execution"
            )

        self.enrich_runtime(runtime, project)
        run = self._create_run_object(task)

        if self._is_run_local:
            runtime = self._create_local_function_for_execution(
                runtime=runtime,
                run=run,
                local_code_path=local_code_path,
                project=project,
                name=name,
                workdir=workdir,
                handler=handler,
                reset_on_run=reset_on_run,
            )

        # sanity check
        elif runtime._is_remote:
            message = "Remote function cannot be executed locally"
            logger.error(
                message,
                is_remote=runtime._is_remote,
                local=self._is_run_local,
                runtime=runtime.to_dict(),
            )
            raise mlrun.errors.MLRunRuntimeError(message)

        run = self._enrich_run(
            runtime=runtime,
            run=run,
            handler=handler,
            project_name=project,
            name=name,
            params=params,
            inputs=inputs,
            returns=returns,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            out_path=out_path,
            artifact_path=artifact_path,
            workdir=workdir,
            notifications=notifications,
            state_thresholds=state_thresholds,
        )
        self._validate_runtime(runtime, run)
        result = self._execute(
            runtime=runtime,
            run=run,
        )

        return result

    def _execute(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        run: Optional[Union["mlrun.run.RunTemplate", "mlrun.run.RunObject"]] = None,
    ):
        if (
            "V3IO_USERNAME" in os.environ
            and mlrun_constants.MLRunInternalLabels.v3io_user not in run.metadata.labels
        ):
            run.metadata.labels[mlrun_constants.MLRunInternalLabels.v3io_user] = (
                os.environ.get("V3IO_USERNAME")
            )

        # store function object in db unless running from within a run pod
        if not runtime.is_child:
            logger.info(
                "Storing function",
                name=run.metadata.name,
                uid=run.metadata.uid,
                db=runtime.spec.rundb,
            )
            self._store_function(runtime, run)

        execution = mlrun.run.MLClientCtx.from_dict(
            run.to_dict(),
            runtime._get_db(),
            autocommit=False,
            is_api=False,
            store_run=False,
        )

        # create task generator (for child runs) from spec
        task_generator = mlrun.runtimes.generators.get_generator(run.spec, execution)
        if task_generator:
            # verify valid task parameters
            tasks = task_generator.generate(run)
            for task in tasks:
                self._validate_run_params(task.spec.parameters)

        # post verifications, store execution in db and run pre run hooks
        execution.store_run()
        runtime._pre_run(run, execution)  # hook for runtime specific prep

        last_err = None
        # If the runtime is nested, it means the hyper-run will run within a single instance of the run.
        # So while in the API, we consider the hyper-run as a single run, and then in the runtime itself when the
        # runtime is now a local runtime and therefore `self._is_nested == False`, we run each task as a separate run by
        # using the task generator
        # TODO client-server separation might not need the not runtime._is_nested anymore as this executed local func
        if task_generator and not runtime._is_nested:
            # multiple runs (based on hyper params or params file)
            runner = runtime._run_many
            if hasattr(runtime, "_parallel_run_many") and task_generator.use_parallel():
                runner = runtime._parallel_run_many
            results = runner(task_generator, execution, run)
            mlrun.runtimes.utils.results_to_iter(results, run, execution)
            result = execution.to_dict()
            result = runtime._update_run_state(result, task=run)

        else:
            # single run
            try:
                resp = runtime._run(run, execution)
                result = runtime._update_run_state(resp, task=run)
            except mlrun.runtimes.base.RunError as err:
                last_err = err
                result = runtime._update_run_state(task=run, err=err)

        self._push_notifications(run, runtime)

        # run post run hooks
        runtime._post_run(result, execution)  # hook for runtime specific cleanup

        return self._wrap_run_result(runtime, result, run, err=last_err)

    def _create_local_function_for_execution(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        run: "mlrun.run.RunObject",
        local_code_path: Optional[str] = None,
        project: Optional[str] = "",
        name: Optional[str] = "",
        workdir: Optional[str] = "",
        handler: Optional[str] = None,
        reset_on_run: Optional[bool] = None,
    ):
        project = project or runtime.metadata.project
        function_name = name or runtime.metadata.name
        command, args = self._resolve_local_code_path(local_code_path)
        if command:
            function_name = name or pathlib.Path(command).stem

        meta = mlrun.model.BaseMetadata(function_name, project=project)

        command, loaded_runtime = mlrun.run.load_func_code(
            command or runtime, workdir, name=name
        )
        # loaded_runtime is loaded from runtime or yaml file, if passed a command it should be None,
        # so we keep the current runtime for enrichment
        runtime = loaded_runtime or runtime
        if loaded_runtime:
            if run:
                handler = handler or run.spec.handler
            handler = handler or runtime.spec.default_handler or ""
            meta = runtime.metadata.copy()
            meta.name = function_name or meta.name
            meta.project = project or meta.project

        # if the handler has module prefix force "local" (vs "handler") runtime
        kind = "local" if isinstance(handler, str) and "." in handler else ""
        fn = mlrun.new_function(meta.name, command=command, args=args, kind=kind)
        fn.metadata = meta
        setattr(fn, "_is_run_local", True)
        if workdir:
            fn.spec.workdir = str(workdir)

        fn.spec.allow_empty_resources = runtime.spec.allow_empty_resources
        if runtime:
            # copy the code/base-spec to the local function (for the UI and code logging)
            fn.spec.description = runtime.spec.description
            fn.spec.build = runtime.spec.build

        run.spec.handler = handler
        run.spec.reset_on_run = reset_on_run
        return fn

    @staticmethod
    def _resolve_local_code_path(local_code_path: str) -> (str, list[str]):
        command = None
        args = []
        if local_code_path:
            command = local_code_path
            if command:
                sp = command.split()
                # split command and args
                command = sp[0]
                if len(sp) > 1:
                    args = sp[1:]
        return command, args

    def _push_notifications(
        self, runobj: "mlrun.run.RunObject", runtime: "mlrun.runtimes.BaseRuntime"
    ):
        if not self._run_has_valid_notifications(runobj):
            return
        # TODO: add store_notifications API endpoint so we can store notifications pushed from the
        #       SDK for documentation purposes.
        # The run is local, so we can assume that watch=True, therefore this code runs
        # once the run is completed, and we can just push the notifications.
        # Only push from jupyter, not from the CLI.
        # "handler" and "dask" kinds are special cases of local runs which don't set local=True
        if self._is_run_local or runtime.kind in ["handler"]:
            mlrun.utils.notifications.NotificationPusher([runobj]).push()
        elif runtime.kind in ["dask"]:
            runtime._get_db().push_run_notifications(
                uid=runobj.metadata.uid, project=runobj.metadata.project
            )
