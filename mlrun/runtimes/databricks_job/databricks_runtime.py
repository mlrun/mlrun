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

import os
from base64 import b64decode, b64encode
from typing import Callable, Dict, List, Optional, Union

from kubernetes.client import V1ExecAction, V1Handler, V1Lifecycle

import mlrun
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.model import HyperParamOptions, RunObject
from mlrun.runtimes.kubejob import KubejobRuntime


class DatabricksRuntime(KubejobRuntime):
    kind = "databricks"
    _is_remote = True

    @staticmethod
    def _get_lifecycle():
        script_path = "/mlrun/mlrun/runtimes/databricks_job/databricks_cancel_task.py"
        pre_stop_handler = V1Handler(
            _exec=V1ExecAction(command=["python", script_path])
        )
        return V1Lifecycle(pre_stop=pre_stop_handler)

    def get_internal_parameters(self, runobj: RunObject):
        """
        Return the internal function code.
        """
        task_parameters = runobj.spec.parameters.get("task_parameters", {})
        if "original_handler" in task_parameters:
            original_handler = task_parameters["original_handler"]
        else:
            original_handler = runobj.spec.handler or ""
        encoded_code = (
            self.spec.build.functionSourceCode if hasattr(self.spec, "build") else None
        )
        if not encoded_code:
            return "", original_handler
        decoded_code = b64decode(encoded_code).decode("utf-8")
        code = _databricks_script_code + decoded_code
        if original_handler:
            code += f"\n{original_handler}(**handler_arguments)\n"
        code = b64encode(code.encode("utf-8")).decode("utf-8")
        return code, original_handler

    def _pre_run(self, runspec: RunObject, execution):
        internal_code, original_handler = self.get_internal_parameters(runspec)
        if internal_code:
            task_parameters = runspec.spec.parameters.get("task_parameters", {})
            task_parameters["spark_app_code"] = internal_code
            if original_handler:
                task_parameters[
                    "original_handler"
                ] = original_handler  # in order to handle reruns.
            runspec.spec.parameters["task_parameters"] = task_parameters
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            databricks_runtime_wrap_path = os.path.join(
                current_dir, "databricks_wrapper.py"
            )
            with open(
                databricks_runtime_wrap_path, "r"
            ) as databricks_runtime_wrap_file:
                wrap_code = databricks_runtime_wrap_file.read()
                wrap_code = b64encode(wrap_code.encode("utf-8")).decode("utf-8")
            self.spec.build.functionSourceCode = wrap_code
            runspec.spec.handler = "run_mlrun_databricks_job"
        else:
            raise ValueError("Databricks function must be provided with user code")

    def run(
        self,
        runspec: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[Union[str, Callable]] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[Dict[str, str]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        watch: Optional[bool] = True,
        schedule: Optional[Union[str, mlrun.common.schemas.ScheduleCronTrigger]] = None,
        hyperparams: Optional[Dict[str, list]] = None,
        hyper_param_options: Optional[HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local: Optional[bool] = False,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[Dict[str, str]] = None,
        notifications: Optional[List[mlrun.model.Notification]] = None,
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
        **launcher_kwargs,
    ) -> RunObject:
        if local:
            raise MLRunInvalidArgumentError("Databricks runtime cannot run locally.")
        return super().run(
            runspec=runspec,
            handler=handler,
            name=name,
            project=project,
            params=params,
            inputs=inputs,
            out_path=out_path,
            workdir=workdir,
            artifact_path=artifact_path,
            watch=watch,
            schedule=schedule,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            local=local,
            local_code_path=local_code_path,
            auto_build=auto_build,
            param_file_secrets=param_file_secrets,
            notifications=notifications,
            returns=returns,
            **launcher_kwargs,
        )


_databricks_script_code = """

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('handler_arguments')
handler_arguments = parser.parse_args().handler_arguments
handler_arguments = json.loads(handler_arguments)

"""
