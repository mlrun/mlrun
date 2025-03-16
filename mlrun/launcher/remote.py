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
from typing import Optional, Union

import pandas as pd
import requests

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.schedule
import mlrun.db
import mlrun.errors
import mlrun.launcher.client as launcher
import mlrun.run
import mlrun.runtimes
import mlrun.runtimes.generators
import mlrun.utils.clones
import mlrun.utils.notifications
from mlrun.utils import logger


class ClientRemoteLauncher(launcher.ClientBaseLauncher):
    def launch(
        self,
        runtime: "mlrun.runtimes.KubejobRuntime",
        task: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[str] = None,
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
        self.enrich_runtime(runtime, project)
        run = self._create_run_object(task)

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

        if not runtime.is_deployed():
            if runtime.spec.build.auto_build or auto_build:
                logger.info(
                    "Function is not deployed and auto_build flag is set, starting deploy..."
                )
                runtime.deploy(skip_deployed=True, show_on_failure=True)

            else:
                if runtime.requires_build():
                    logger.warning(
                        "Function image is not built/ready and function requires build - execution will fail. "
                        "Need to set auto_build=True or use .deploy() method first"
                    )

        if runtime.verbose:
            logger.info(f"runspec:\n{run.to_yaml()}")

        if (
            "V3IO_USERNAME" in os.environ
            and mlrun_constants.MLRunInternalLabels.v3io_user not in run.metadata.labels
        ):
            run.metadata.labels[mlrun_constants.MLRunInternalLabels.v3io_user] = (
                os.environ.get("V3IO_USERNAME")
            )

        logger.info(
            "Storing function",
            name=run.metadata.name,
            uid=run.metadata.uid,
            db=runtime.spec.rundb,
        )
        self._store_function(runtime, run)

        return self._submit_job(runtime, run, schedule, watch)

    def _submit_job(
        self,
        runtime: "mlrun.runtimes.KubejobRuntime",
        run: "mlrun.run.RunObject",
        schedule: Optional[mlrun.common.schemas.ScheduleCronTrigger] = None,
        watch: Optional[bool] = None,
    ):
        if runtime._secrets:
            run.spec.secret_sources = runtime._secrets.to_serial()
        try:
            db = runtime._get_db()
            resp = db.submit_job(run, schedule=schedule)
            if schedule:
                action = resp.pop("action", "created")
                logger.info(f"Task schedule {action}", **resp)
                return

        except (requests.HTTPError, Exception) as err:
            logger.error("Failed remote run", error=mlrun.errors.err_to_str(err))

            if isinstance(err, requests.HTTPError):
                runtime._handle_submit_job_http_error(err)

            result = None
            # if we got a schedule no reason to do post_run stuff (it purposed to update the run status with error,
            # but there's no run in case of schedule)
            if not schedule:
                result = runtime._update_run_state(
                    task=run, err=mlrun.errors.err_to_str(err)
                )
            return self._wrap_run_result(
                runtime, result, run, schedule=schedule, err=err
            )

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
            run.logs(True, runtime._get_db())
            resp = runtime._get_db_run(run)

        return self._wrap_run_result(runtime, resp, run, schedule=schedule)

    @classmethod
    def _validate_run_single_param(cls, param_name, param_value):
        if isinstance(param_value, pd.DataFrame):
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"Parameter '{param_name}' has an unsupported value of type"
                f" 'pandas.DataFrame' in remote execution."
            )
        super()._validate_run_single_param(
            param_name=param_name, param_value=param_value
        )
