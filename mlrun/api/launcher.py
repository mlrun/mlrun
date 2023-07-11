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
from typing import Dict, List, Optional, Union

import mlrun.api.crud
import mlrun.common.db.sql_session
import mlrun.common.schemas.schedule
import mlrun.config
import mlrun.execution
import mlrun.launcher.base as launcher
import mlrun.runtimes
import mlrun.runtimes.generators
import mlrun.runtimes.utils
import mlrun.utils
import mlrun.utils.regex


class ServerSideLauncher(launcher.BaseLauncher):
    def launch(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        task: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[str] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[Dict[str, str]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        watch: Optional[bool] = True,
        schedule: Optional[
            Union[str, mlrun.common.schemas.schedule.ScheduleCronTrigger]
        ] = None,
        hyperparams: Dict[str, list] = None,
        hyper_param_options: Optional[mlrun.model.HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[Dict[str, str]] = None,
        notifications: Optional[List[mlrun.model.Notification]] = None,
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> mlrun.run.RunObject:
        self.enrich_runtime(runtime, project)

        run = self._create_run_object(task)

        run = self._enrich_run(
            runtime,
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
        )
        self._validate_runtime(runtime, run)

        if runtime.verbose:
            mlrun.utils.logger.info(f"Run:\n{run.to_yaml()}")

        if not runtime.is_child:
            mlrun.utils.logger.info(
                "Storing function",
                name=run.metadata.name,
                uid=run.metadata.uid,
            )
            self._store_function(runtime, run)

        execution = mlrun.execution.MLClientCtx.from_dict(
            run.to_dict(),
            runtime._get_db(),
            autocommit=False,
            is_api=True,
            store_run=False,
        )

        # create task generator (for child runs) from spec
        task_generator = mlrun.runtimes.generators.get_generator(
            run.spec, execution, param_file_secrets=param_file_secrets
        )
        if task_generator:
            # verify valid task parameters
            tasks = task_generator.generate(run)
            for task in tasks:
                self._validate_run_params(task.spec.parameters)

        # post verifications, store execution in db and run pre run hooks
        execution.store_run()
        runtime._pre_run(run, execution)  # hook for runtime specific prep

        resp = None
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

            except mlrun.runtimes.utils.RunError as err:
                last_err = err

            finally:
                result = runtime._update_run_state(resp=resp, task=run, err=last_err)

        self._save_notifications(run)

        runtime._post_run(result, execution)  # hook for runtime specific cleanup

        return self._wrap_run_result(runtime, result, run, err=last_err)

    @staticmethod
    def enrich_runtime(
        runtime: "mlrun.runtimes.base.BaseRuntime", project_name: Optional[str] = ""
    ):
        """
        Enrich the runtime object with the project spec and metadata.
        This is done only on the server side, since it's the source of truth for the project, and we want to keep the
        client side enrichment as minimal as possible.
        """
        # ensure the runtime has a project before we enrich it with the project's spec
        runtime.metadata.project = (
            project_name
            or runtime.metadata.project
            or mlrun.config.config.default_project
        )
        project = runtime._get_db().get_project(runtime.metadata.project)
        # this is mainly for tests with nop db
        # in normal use cases if no project is found we will get an error
        if project:
            project = mlrun.projects.project.MlrunProject.from_dict(project.dict())
            mlrun.projects.pipelines.enrich_function_object(
                project, runtime, copy_function=False
            )

    def _save_notifications(self, runobj):
        if not self._run_has_valid_notifications(runobj):
            return

        # If in the api server, we can assume that watch=False, so we save notification
        # configs to the DB, for the run monitor to later pick up and push.
        session = mlrun.common.db.session.create_session()
        mlrun.api.crud.Notifications().store_run_notifications(
            session,
            runobj.spec.notifications,
            runobj.metadata.uid,
            runobj.metadata.project,
        )

    def _store_function(
        self, runtime: mlrun.runtimes.base.BaseRuntime, run: mlrun.run.RunObject
    ):
        run.metadata.labels["kind"] = runtime.kind
        db = runtime._get_db()
        if db and runtime.kind != "handler":
            struct = runtime.to_dict()
            hash_key = db.store_function(
                struct, runtime.metadata.name, runtime.metadata.project, versioned=True
            )
            run.spec.function = runtime._function_uri(hash_key=hash_key)
