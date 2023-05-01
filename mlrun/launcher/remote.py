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
import typing
from typing import Dict, List, Union

import mlrun.runtimes
from mlrun.launcher.base import BaseLauncher


class ClientRemoteLauncher(BaseLauncher):
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
        returns: typing.Optional[List[Union[str, Dict[str, str]]]] = None,
    ):
        pass

    @staticmethod
    def _enrich_runtime(runtime):
        pass

    @staticmethod
    def _validate_runtime(runtime):
        pass

    def _submit_job(self, runtime):
        pass

    def _deploy(self, runtime):
        pass
