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

import mlrun.api.crud
import mlrun.api.db.sqldb.session
import mlrun.model
import mlrun.runtimes.generators
import mlrun.runtimes.utils
import mlrun.utils.regex
from mlrun.execution import MLClientCtx
from mlrun.launcher.base import BaseLauncher
from mlrun.model import RunObject
from mlrun.runtimes import BaseRuntime
from mlrun.utils import logger


class ServerSideLauncher(BaseLauncher):
    def __init__(self):
        self._db_conn = None

    def launch(self, runtime: BaseRuntime, run: RunObject, param_file_secrets):
        self._enrich_run(runtime, run)
        if runtime.verbose:
            logger.info(f"Run:\n{run.to_yaml()}")

        db = self._get_db(runtime)
        if not runtime.is_child:
            logger.info(
                "Storing function",
                name=run.metadata.name,
                uid=run.metadata.uid,
            )
            self._store_function(runtime, run, db)

        execution = MLClientCtx.from_dict(
            run.to_dict(),
            db,
            autocommit=False,
            is_api=True,
            store_run=False,
        )

        self._verify_run_params(run.spec.parameters)

        # create task generator (for child runs) from spec
        task_generator = mlrun.runtimes.generators.get_generator(
            run.spec, execution, param_file_secrets=param_file_secrets
        )
        if task_generator:
            # verify valid task parameters
            tasks = task_generator.generate(run)
            for task in tasks:
                self._verify_run_params(task.spec.parameters)

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

        self._save_or_push_notifications(run)

        runtime._post_run(result, execution)  # hook for runtime specific cleanup

        return runtime._wrap_run_result(result, run, err=last_err)

    @staticmethod
    def verify_base_image(runtime):
        pass

    @staticmethod
    def save(runtime):
        pass

    @staticmethod
    def _enrich_runtime(runtime):
        """
        Enrich the function with:
            1. Default values
            2. mlrun config values
            3. Project context values
            4. Run specific parameters
        """
        pass

    @staticmethod
    def _save_or_push_notifications(run: RunObject):
        if not run.spec.notifications:
            logger.debug("No notifications to push for run", run_uid=run.metadata.uid)
            return

        # TODO: add support for other notifications per run iteration
        if run.metadata.iteration and run.metadata.iteration > 0:
            logger.debug(
                "Notifications per iteration are not supported, skipping",
                run_uid=run.metadata.uid,
            )
            return

        # If in the api server, we can assume that watch=False, so we save notification
        # configs to the DB, for the run monitor to later pick up and push.
        session = mlrun.api.db.sqldb.session.create_session()
        mlrun.api.crud.Notifications().store_run_notifications(
            session,
            run.spec.notifications,
            run.metadata.uid,
            run.metadata.project,
        )

    @staticmethod
    def _ensure_run_db(runtime: BaseRuntime):
        runtime.spec.rundb = runtime.spec.rundb or mlrun.db.get_or_set_dburl()

    def _get_db(self, runtime: BaseRuntime):
        self._ensure_run_db(runtime)
        if not self._db_conn:
            if runtime.spec.rundb:
                self._db_conn = mlrun.db.get_run_db(
                    runtime.spec.rundb, secrets=runtime._secrets
                )
        return self._db_conn

    @staticmethod
    def _store_function(runtime: BaseRuntime, run: RunObject, db):
        run.metadata.labels["kind"] = runtime.kind
        if db and runtime.kind != "handler":
            struct = runtime.to_dict()
            hash_key = db.store_function(
                struct, runtime.metadata.name, runtime.metadata.project, versioned=True
            )
            run.spec.function = runtime._function_uri(hash_key=hash_key)
