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
import os
import pathlib
from typing import Dict, List, Optional, Union

import mlrun.errors
from mlrun.launcher.base import BaseLauncher
from mlrun.utils import logger


class ClientLocalLauncher(BaseLauncher):
    @staticmethod
    def verify_base_image(runtime):
        pass

    @staticmethod
    def save(runtime):
        pass

    def run(
        self,
        runtime,
        runspec=None,
        handler=None,
        name: str = "",
        project: str = "",
        params: dict = None,
        inputs: Dict[str, str] = None,
        out_path: str = "",
        workdir: str = "",
        artifact_path: str = "",
        watch: bool = True,
        # TODO: don't use schedule from API schemas but rather from mlrun client
        schedule=None,  # Union[str, mlrun.api.schemas.schedule.ScheduleCronTrigger]
        hyperparams: Dict[str, list] = None,
        hyper_param_options=None,  # :mlrun.model.HyperParamOptions
        verbose=None,
        scrape_metrics: bool = None,
        local=False,
        local_code_path=None,
        auto_build=None,
        param_file_secrets: Dict[str, str] = None,
        notifications=None,  # : List[mlrun.model.Notification]
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ):
        self._enrich_runtime(runtime)

        run = runtime._create_run_object(runspec)

        # do not allow local function to be scheduled
        if schedule is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "local and schedule cannot be used together"
            )
        result = self._run_local(
            runtime,
            run,
            local_code_path,
            project,
            name,
            workdir,
            handler,
            params,
            inputs,
            returns,
            artifact_path,
            notifications=notifications,
        )

        runtime._save_or_push_notifications(result, local)
        return result

    @staticmethod
    def _enrich_runtime(runtime):
        runtime.try_auto_mount_based_on_config()
        runtime._fill_credentials()

    @staticmethod
    def _validate_runtime(runtime):
        pass

    def _run_local(
        self,
        runtime,
        runspec,
        local_code_path,
        project,
        name,
        workdir,
        handler,
        params,
        inputs,
        returns,
        artifact_path,
        notifications=None,  # : List[mlrun.model.Notification]
    ):
        command = runtime
        if local_code_path:
            project = project or runtime.metadata.project
            name = name or runtime.metadata.name
            command = local_code_path
        return self.run_local(
            runspec,
            command,
            name,
            runtime.spec.args,
            workdir=workdir,
            project=project,
            handler=handler,
            params=params,
            inputs=inputs,
            artifact_path=artifact_path,
            mode=runtime.spec.mode,
            allow_empty_resources=runtime.spec.allow_empty_resources,
            notifications=notifications,
            returns=returns,
        )

    def run_local(
        self,
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
        notifications=None,  # : List[mlrun.model.Notification]
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
        :param returns:  List of configurations for how to log the returning values from the handler's run
                        (as artifacts or results). The list's length must be equal to the amount of returning objects.
                         A configuration may be given as:

                         * A string of the key to use to log the returning value as result or as an artifact. To specify
                           The artifact type, it is possible to pass a string in the following structure:
                           "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`.
                           If no artifact type is specified, the object's default artifact type will be used.
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

        meta = mlrun.model.BaseMetadata(function_name, project=project, tag=tag)
        from mlrun.run import load_func_code

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
        fn = mlrun.new_function(
            meta.name, command=command, args=args, mode=mode, kind=kind
        )
        fn.metadata = meta
        setattr(fn, "_is_run_local", True)
        if workdir:
            fn.spec.workdir = str(workdir)
        fn.spec.allow_empty_resources = allow_empty_resources
        if runtime:
            # copy the code/base-spec to the local function (for the UI and code logging)
            fn.spec.description = runtime.spec.description
            fn.spec.build = runtime.spec.build
        return self._run(
            runtime=fn,
            runspec=task,
            handler=handler,
            name=name,
            params=params,
            inputs=inputs,
            returns=returns,
            artifact_path=artifact_path,
            notifications=notifications,
        )

    def _run(
        self,
        runtime,
        runspec=None,
        handler=None,
        name: str = "",
        project: str = "",
        params: dict = None,
        inputs: Dict[str, str] = None,
        out_path: str = "",
        workdir: str = "",
        artifact_path: str = "",
        watch: bool = True,
        # TODO: don't use schedule from API schemas but rather from mlrun client
        schedule=None,  # : Union[str, mlrun.api.schemas.schedule.ScheduleCronTrigger]
        hyperparams: Dict[str, list] = None,
        hyper_param_options=None,  # : mlrun.model.HyperParamOptions
        verbose=None,
        scrape_metrics: bool = None,
        param_file_secrets: Dict[str, str] = None,
        notifications=None,  # : List[mlrun.model.Notification]
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ):
        run = runtime._enrich_run(
            runspec,
            handler,
            project,
            name,
            params,
            inputs,
            returns,
            hyperparams,
            hyper_param_options,
            verbose,
            scrape_metrics,
            out_path,
            artifact_path,
            workdir,
            notifications,
        )
        runtime._validate_output_path(run)
        db = runtime._get_db()

        if runtime.verbose:
            logger.info(f"runspec:\n{run.to_yaml()}")

        if "V3IO_USERNAME" in os.environ and "v3io_user" not in run.metadata.labels:
            run.metadata.labels["v3io_user"] = os.environ.get("V3IO_USERNAME")

        logger.info(
            "Storing function",
            name=run.metadata.name,
            uid=run.metadata.uid,
            db=runtime.spec.rundb,
        )
        runtime._store_function(run, run.metadata, db)

        from mlrun.run import MLClientCtx

        execution = MLClientCtx.from_dict(
            run.to_dict(),
            db,
            autocommit=False,
            is_api=False,
            store_run=False,
        )

        runtime._verify_run_params(run.spec.parameters)
        from mlrun.runtimes.generators import get_generator

        # create task generator (for child runs) from spec
        task_generator = get_generator(
            run.spec, execution, param_file_secrets=param_file_secrets
        )
        if task_generator:
            # verify valid task parameters
            tasks = task_generator.generate(run)
            for task in tasks:
                runtime._verify_run_params(task.spec.parameters)

        # post verifications, store execution in db and run pre run hooks
        execution.store_run()
        runtime._pre_run(run, execution)  # hook for runtime specific prep

        last_err = None
        # If the runtime is nested, it means the hyper-run will run within a single instance of the run.
        # So while in the API, we consider the hyper-run as a single run, and then in the runtime itself when the
        # runtime is now a local runtime and therefore `self._is_nested == False`, we run each task as a separate run by
        # using the task generator
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
                if (
                    watch
                    and mlrun.runtimes.RuntimeKinds.is_watchable(runtime.kind)
                    # API shouldn't watch logs, it's the client job to query the run logs
                    and not mlrun.config.is_running_as_api()
                ):
                    state, _ = run.logs(True, runtime._get_db())
                    if state not in ["succeeded", "completed"]:
                        logger.warning(f"run ended with state {state}")
                result = runtime._update_run_state(resp, task=run)
            except mlrun.runtimes.base.RunError as err:
                last_err = err
                result = runtime._update_run_state(task=run, err=err)

        self._save_or_push_notifications(run)
        # run post run hooks
        runtime._post_run(result, execution)  # hook for runtime specific cleanup

        return runtime._wrap_run_result(result, run, schedule=schedule, err=last_err)

    def _save_or_push_notifications(self, runobj):
        from mlrun.utils.notifications import NotificationPusher

        if not self._are_validate_notifications(runobj):
            return
        # The run is local, so we can assume that watch=True, therefore this code runs
        # once the run is completed, and we can just push the notifications.
        # TODO: add store_notifications API endpoint so we can store notifications pushed from the
        #       SDK for documentation purposes.
        NotificationPusher([runobj]).push()
