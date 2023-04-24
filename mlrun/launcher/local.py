# Copyright 2023 MLRun Authors
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
import getpass
import os
import pathlib
import typing
from typing import Dict, List, Optional, Union

import mlrun.api.schemas.schedule
import mlrun.db
import mlrun.errors
import mlrun.run
import mlrun.runtimes.generators
import mlrun.utils.clones
import mlrun.utils.notifications
from mlrun.launcher.base import BaseLauncher
from mlrun.utils import logger


class ClientLocalLauncher(BaseLauncher):
    @staticmethod
    def verify_base_image(runtime):
        pass

    @staticmethod
    def save(runtime):
        pass

    def launch(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        task: typing.Optional[
            typing.Union[mlrun.run.RunTemplate, mlrun.run.RunObject]
        ] = None,
        handler: typing.Optional[str] = None,
        name: typing.Optional[str] = "",
        project: typing.Optional[str] = "",
        params: typing.Optional[dict] = None,
        inputs: typing.Optional[Dict[str, str]] = None,
        out_path: typing.Optional[str] = "",
        workdir: typing.Optional[str] = "",
        artifact_path: typing.Optional[str] = "",
        watch: typing.Optional[bool] = True,
        # TODO: don't use schedule from API schemas but rather from mlrun client
        schedule: typing.Optional[
            typing.Union[str, mlrun.api.schemas.schedule.ScheduleCronTrigger]
        ] = None,
        hyperparams: Dict[str, list] = None,
        hyper_param_options: typing.Optional[
            mlrun.model.HyperParamOptions
        ] = None,  # :mlrun.model.HyperParamOptions
        verbose: typing.Optional[bool] = None,
        scrape_metrics: typing.Optional[bool] = None,
        local: typing.Optional[bool] = False,
        local_code_path: typing.Optional[str] = None,
        auto_build: typing.Optional[bool] = None,
        param_file_secrets: typing.Optional[Dict[str, str]] = None,
        notifications: typing.Optional[List[mlrun.model.Notification]] = None,
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ):
        # do not allow local function to be scheduled
        if schedule is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "local and schedule cannot be used together"
            )

        self._enrich_runtime(runtime)

        run = self._create_run_object(task)

        local_function = self._create_local_function_for_execution(
            runtime=runtime,
            run=run,
            local_code_path=local_code_path,
            project=project,
            name=name,
            workdir=workdir,
            handler=handler,
        )

        result = self.execute(
            runtime=local_function,
            task=run,
            name=name,
            params=params,
            inputs=inputs,
            returns=returns,
            artifact_path=artifact_path,
            notifications=notifications,
        )

        self._save_or_push_notifications(result)
        return result

    @staticmethod
    def _enrich_runtime(runtime):
        runtime.try_auto_mount_based_on_config()
        runtime._fill_credentials()

    @staticmethod
    def _validate_runtime(runtime):
        pass

    def _create_local_function_for_execution(
        self,
        runtime,
        run,
        local_code_path,
        project,
        name,
        workdir,
        handler,
    ):

        project = project or runtime.metadata.project
        function_name = name or runtime.metadata.name
        command, args = self._resolve_local_code_path(local_code_path)
        if command:
            function_name = name or pathlib.Path(command).stem

        meta = mlrun.model.BaseMetadata(function_name, project=project)

        command, runtime = mlrun.run.load_func_code(
            command or runtime, workdir, name=name
        )
        if runtime:
            if run:
                handler = handler or run.spec.handler
            handler = handler or runtime.spec.default_handler or ""
            meta = runtime.metadata.copy()
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
        return fn

    @staticmethod
    def _resolve_local_code_path(local_code_path: str) -> (str, typing.List[str]):
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

    def execute(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        task: typing.Optional[
            typing.Union[mlrun.run.RunTemplate, mlrun.run.RunObject]
        ] = None,
        name: typing.Optional[str] = "",
        project: typing.Optional[str] = "",
        params: typing.Optional[dict] = None,
        inputs: typing.Optional[Dict[str, str]] = None,
        artifact_path: typing.Optional[str] = "",
        notifications: typing.Optional[List[mlrun.model.Notification]] = None,
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ):
        run = self._enrich_run(
            runtime=runtime,
            runspec=task,
            project_name=project,
            name=name,
            params=params,
            inputs=inputs,
            returns=returns,
            artifact_path=artifact_path,
            notifications=notifications,
        )
        # TODO client-server separation, combine all validation under one method
        runtime._validate_output_path(run)
        runtime._validate_run_params(run.spec.parameters)

        if "V3IO_USERNAME" in os.environ and "v3io_user" not in run.metadata.labels:
            run.metadata.labels["v3io_user"] = os.environ.get("V3IO_USERNAME")

        logger.info(
            "Storing function",
            name=run.metadata.name,
            uid=run.metadata.uid,
            db=runtime.spec.rundb,
        )
        self.store_function(runtime, run, run.metadata)

        execution = mlrun.run.MLClientCtx.from_dict(
            run.to_dict(),
            self.db,
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
                runtime._validate_run_params(task.spec.parameters)

        # post verifications, store execution in db and run pre run hooks
        execution.store_run()
        self._pre_run(runtime, run, execution)  # hook for runtime specific prep

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

        self._save_or_push_notifications(run)
        # run post run hooks
        self._post_run(execution)  # hook for runtime specific cleanup

        return runtime._wrap_run_result(result, run, err=last_err)

    @staticmethod
    def _pre_run(runtime, runobj, execution):
        workdir = runtime.spec.workdir
        execution._current_workdir = workdir
        execution._old_workdir = None

        # TODO client-server, might not need the _is_run_local any more
        if runtime.spec.build.source and not hasattr(runtime, "_is_run_local"):
            target_dir = mlrun.utils.clones.extract_source(
                runtime.spec.build.source,
                runtime.spec.clone_target_dir,
                secrets=execution._secrets_manager,
            )
            if workdir and not workdir.startswith("/"):
                execution._current_workdir = os.path.join(target_dir, workdir)
            else:
                execution._current_workdir = workdir or target_dir

        if execution._current_workdir:
            execution._old_workdir = os.getcwd()
            workdir = os.path.realpath(execution._current_workdir)
            mlrun.utils.helpers.set_paths(workdir)
            os.chdir(workdir)
        else:
            mlrun.utils.helpers.set_paths(os.path.realpath("."))

        if (
            runobj.metadata.labels.get("kind") == mlrun.runtimes.RemoteSparkRuntime.kind
            and os.environ["MLRUN_SPARK_CLIENT_IGZ_SPARK"] == "true"
        ):
            mlrun.runtimes.remotesparkjob.igz_spark_pre_hook()

    @staticmethod
    def _post_run(execution):
        # hook for runtime specific cleanup
        if execution._old_workdir:
            os.chdir(execution._old_workdir)

    def store_function(self, runtime, runspec, meta):
        meta.labels["kind"] = runtime.kind
        if "owner" not in meta.labels:
            meta.labels["owner"] = os.environ.get("V3IO_USERNAME") or getpass.getuser()
        if runspec.spec.output_path:
            runspec.spec.output_path = runspec.spec.output_path.replace(
                "{{run.user}}", meta.labels["owner"]
            )

        if self.db and runtime.kind != "handler":
            struct = runtime.to_dict()
            hash_key = self.db.store_function(
                struct, runtime.metadata.name, runtime.metadata.project, versioned=True
            )
            runspec.spec.function = runtime._function_uri(hash_key=hash_key)

    def _save_or_push_notifications(self, runobj):
        if not self._are_validate_notifications(runobj):
            return
        # The run is local, so we can assume that watch=True, therefore this code runs
        # once the run is completed, and we can just push the notifications.
        # TODO: add store_notifications API endpoint so we can store notifications pushed from the
        #       SDK for documentation purposes.
        mlrun.utils.notifications.NotificationPusher([runobj]).push()
