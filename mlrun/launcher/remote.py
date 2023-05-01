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
import mlrun.runtimes
import os
import typing
from typing import Dict, List, Optional, Union

import requests

import mlrun.api.schemas.schedule
import mlrun.db
import mlrun.db.httpdb
import mlrun.errors
import mlrun.run
import mlrun.runtimes.generators
import mlrun.utils.clones
import mlrun.utils.notifications
from mlrun.launcher.base import BaseLauncher
from mlrun.utils import logger

# TODO client-server-separation: share between all launchers
run_modes = ["pass"]


class ClientRemoteLauncher(BaseLauncher):
    @property
    def db(self) -> mlrun.db.httpdb.HTTPRunDB:
        return self._db

    def _save_or_push_notifications(self, runobj):
        pass

    @staticmethod
    def verify_base_image(runtime):
        pass

    @staticmethod
    def save(runtime):
        pass

    def launch(
        self,
        runtime: mlrun.runtimes.KubejobRuntime,
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
        hyper_param_options: typing.Optional[mlrun.model.HyperParamOptions] = None,
        verbose: typing.Optional[bool] = None,
        scrape_metrics: typing.Optional[bool] = None,
        local: typing.Optional[bool] = False,
        local_code_path: typing.Optional[str] = None,
        auto_build: typing.Optional[bool] = None,
        param_file_secrets: typing.Optional[Dict[str, str]] = None,
        notifications: typing.Optional[List[mlrun.model.Notification]] = None,
        returns: typing.Optional[List[Union[str, Dict[str, str]]]] = None,
    ):
        mlrun.utils.helpers.verify_dict_items_type("Inputs", inputs, [str], [str])

        if runtime.spec.mode and runtime.spec.mode not in run_modes:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f'run mode can only be {",".join(run_modes)}'
            )

        self._enrich_runtime(runtime)

        run = self._create_run_object(task)

        run = self._enrich_run(
            runtime=runtime,
            runspec=run,
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

        runtime._validate_output_path(run)

        if not runtime.is_deployed():
            if runtime.spec.build.auto_build or auto_build:
                logger.info(
                    "Function is not deployed and auto_build flag is set, starting deploy..."
                )
                runtime.deploy(skip_deployed=True, show_on_failure=True)

            else:
                raise mlrun.errors.MLRunRuntimeError(
                    "function image is not built/ready, set auto_build=True or use .deploy() method first"
                )

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
        self.store_function(runtime, run)

        return self.submit_job(runtime, run, schedule, watch)

    @staticmethod
    def _enrich_runtime(runtime):
        runtime.try_auto_mount_based_on_config()
        runtime._fill_credentials()

    @staticmethod
    def _validate_runtime(runtime):
        pass

    def submit_job(
        self,
        runtime: mlrun.runtimes.KubejobRuntime,
        run: mlrun.run.RunObject,
        schedule: typing.Optional[mlrun.api.schemas.ScheduleCronTrigger] = None,
        watch: typing.Optional[bool] = None,
    ):
        if runtime._secrets:
            run.spec.secret_sources = runtime._secrets.to_serial()
        try:
            resp = self.db.submit_job(run, schedule=schedule)
            if schedule:
                action = resp.pop("action", "created")
                logger.info(f"task schedule {action}", **resp)
                return

        except (requests.HTTPError, Exception) as err:
            logger.error(f"got remote run err, {mlrun.errors.err_to_str(err)}")

            if isinstance(err, requests.HTTPError):
                runtime._handle_submit_job_http_error(err)

            result = None
            # if we got a schedule no reason to do post_run stuff (it purposed to update the run status with error,
            # but there's no run in case of schedule)
            if not schedule:
                result = runtime._update_run_state(
                    task=run, err=mlrun.errors.err_to_str(err)
                )
            return runtime._wrap_run_result(result, run, schedule=schedule, err=err)

        if resp:
            txt = mlrun.runtimes.utils.helpers.get_in(resp, "status.status_text")
            if txt:
                logger.info(txt)
        # watch is None only in scenario where we run from pipeline step, in this case we don't want to watch the run
        # logs too frequently but rather just pull the state of the run from the DB and pull the logs every x seconds
        # which ideally greater than the pull state interval, this reduces unnecessary load on the API server, as
        # running a pipeline is mostly not an interactive process which means the logs pulling doesn't need to be pulled
        # in real time
        if (
            watch is None
            and runtime.kfp
            and mlrun.mlconf.httpdb.logs.pipelines.pull_state.mode == "enabled"
        ):
            state_interval = int(
                mlrun.mlconf.httpdb.logs.pipelines.pull_state.pull_state_interval
            )
            logs_interval = int(
                mlrun.mlconf.httpdb.logs.pipelines.pull_state.pull_logs_interval
            )

            run.wait_for_completion(
                show_logs=True,
                sleep=state_interval,
                logs_interval=logs_interval,
                raise_on_failure=False,
            )
            resp = runtime._get_db_run(run)

        elif watch or runtime.kfp:
            run.logs(True, self.db)
            resp = runtime._get_db_run(run)

        return runtime._wrap_run_result(resp, run, schedule=schedule)

    def store_function(
        self, runtime: mlrun.runtimes.KubejobRuntime, run: mlrun.run.RunObject
    ):
        metadata = run.metadata
        metadata.labels["kind"] = runtime.kind
        if "owner" not in metadata.labels:
            metadata.labels["owner"] = (
                os.environ.get("V3IO_USERNAME") or getpass.getuser()
            )
        if run.spec.output_path:
            run.spec.output_path = run.spec.output_path.replace(
                "{{run.user}}", metadata.labels["owner"]
            )
        struct = runtime.to_dict()
        hash_key = self.db.store_function(
            struct, runtime.metadata.name, runtime.metadata.project, versioned=True
        )
        run.spec.function = runtime._function_uri(hash_key=hash_key)
