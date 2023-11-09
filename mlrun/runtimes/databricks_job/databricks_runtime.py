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

import mlrun
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.model import HyperParamOptions, RunObject
from mlrun.runtimes.kubejob import KubejobRuntime


class DatabricksRuntime(KubejobRuntime):
    kind = "databricks"
    _is_remote = True

    def _get_log_artifacts_code(self, runobj: RunObject, task_parameters: dict):
        artifact_json_dir = task_parameters.get(
            "artifact_json_dir",
            mlrun.mlconf.function.databricks.artifact_directory_path,
        )
        artifact_json_path = (
            f"{artifact_json_dir}/mlrun_artifact_{runobj.metadata.uid}.json"
        )
        return (
            artifacts_code_template.format(f"/dbfs{artifact_json_path}"),
            artifact_json_path,
        )

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
            raise ValueError("Databricks function must be provided with user code")
        decoded_code = b64decode(encoded_code).decode("utf-8")
        artifacts_code, artifact_json_path = self._get_log_artifacts_code(
            runobj=runobj, task_parameters=task_parameters
        )
        code = artifacts_code + _databricks_script_code + decoded_code
        if original_handler:
            code += f"\nresult = {original_handler}(**handler_arguments)\n"
            code += """\n
default_key_template = 'mlrun_return_value_'
if result:
    if isinstance(result, dict):
        for key, path in result.items():
            mlrun_log_artifact(name=key, path=path)
    elif isinstance(result, (list, tuple, set)):
        for index, value in enumerate(result):
            key = f'{default_key_template}{index+1}'
            mlrun_log_artifact(name=key, path=value)
    elif isinstance(result, str):
        mlrun_log_artifact(name=f'{default_key_template}1', path=result)
    else:
        mlrun_logger.warning(f'cannot log artifacts with the result of handler function \
- result in unsupported type. {type(result)}')
"""
        code = b64encode(code.encode("utf-8")).decode("utf-8")
        updated_task_parameters = {
            "original_handler": original_handler,
            "artifact_json_path": artifact_json_path,
        }
        return code, updated_task_parameters

    def _pre_run(self, runspec: RunObject, execution):
        internal_code, updated_task_parameters = self.get_internal_parameters(runspec)
        task_parameters = runspec.spec.parameters.get("task_parameters", {})
        task_parameters["spark_app_code"] = internal_code
        for key, value in updated_task_parameters.items():
            if value:
                task_parameters[key] = value  # in order to handle reruns.
        runspec.spec.parameters["task_parameters"] = task_parameters
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        databricks_runtime_wrap_path = os.path.join(
            current_dir, "databricks_wrapper.py"
        )
        with open(databricks_runtime_wrap_path, "r") as databricks_runtime_wrap_file:
            wrap_code = databricks_runtime_wrap_file.read()
            wrap_code = b64encode(wrap_code.encode("utf-8")).decode("utf-8")
        self.spec.build.functionSourceCode = wrap_code
        runspec.spec.handler = "run_mlrun_databricks_job"

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

artifacts_code_template = """\n
import logging
mlrun_logger = logging.getLogger('mlrun_logger')
mlrun_logger.setLevel(logging.DEBUG)

def mlrun_log_artifact(name, path):
    if not name or not path:
        mlrun_logger.error(f'name and path required for logging an mlrun artifact - {{name}} : {{path}}')
        return
    if not isinstance(name, str) or not isinstance(path, str):
        mlrun_logger.error(f'name and path must be in string type for logging an mlrun artifact - {{name}} : {{path}}')
        return
    if not path.startswith('/dbfs') and not path.startswith('dbfs:/'):
        mlrun_logger.error(f'path for an mlrun artifact must start with /dbfs or dbfs:/ - {{name}} : {{path}}')
        return
    mlrun_artifacts_path = '{}'
    import json
    import os
    new_data = {{name:path}}
    if os.path.exists(mlrun_artifacts_path):
        with open(mlrun_artifacts_path, 'r+') as json_file:
            existing_data = json.load(json_file)
            existing_data.update(new_data)
            json_file.seek(0)
            json.dump(existing_data, json_file)
    else:
        parent_dir = os.path.dirname(mlrun_artifacts_path)
        if parent_dir != '/dbfs':
            os.makedirs(parent_dir, exist_ok=True)
        with open(mlrun_artifacts_path, 'w') as json_file:
            json.dump(new_data, json_file)
    mlrun_logger.info(f'successfully wrote artifact details to the artifact JSON file in DBFS - {{name}} : {{path}}')
\n
"""
