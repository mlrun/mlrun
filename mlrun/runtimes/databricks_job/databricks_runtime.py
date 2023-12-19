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
from ast import FunctionDef, parse, unparse
from base64 import b64decode, b64encode
from typing import Callable, Dict, List, Optional, Union

import mlrun
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.model import HyperParamOptions, RunObject
from mlrun.runtimes.kubejob import KubejobRuntime


def get_log_artifacts_code(runobj: RunObject, task_parameters: dict):
    artifact_json_dir = task_parameters.get(
        "artifact_json_dir",
        mlrun.mlconf.function.databricks.artifact_directory_path,
    )
    artifact_json_path = (
        f"{artifact_json_dir}/mlrun_artifact_{runobj.metadata.uid}.json"
    )
    return (
        log_artifacts_code_template.format(f"/dbfs{artifact_json_path}"),
        artifact_json_path,
    )


def replace_log_artifact_function(code: str, log_artifacts_code: str):
    #  user can use a dummy function in oder to avoid edit his code.
    #  replace mlrun_log_artifact function if already exist.
    is_replaced = False
    parsed_code = parse(code)
    for node in parsed_code.body:
        if isinstance(node, FunctionDef) and node.name == "mlrun_log_artifact":
            new_function_ast = parse(log_artifacts_code)
            node.args = new_function_ast.body[0].args
            node.body = new_function_ast.body[0].body
            is_replaced = True
            break
    return unparse(parsed_code), is_replaced


class DatabricksRuntime(KubejobRuntime):
    kind = "databricks"
    _is_remote = True

    @staticmethod
    def _verify_returns(returns):
        # TODO complete returns feature
        if returns:
            raise MLRunInvalidArgumentError(
                "Databricks function does not support returns."
            )

    def _get_modified_user_code(self, original_handler: str, log_artifacts_code: str):
        encoded_code = (
            self.spec.build.functionSourceCode if hasattr(self.spec, "build") else None
        )
        if not encoded_code:
            raise ValueError("Databricks function must be provided with user code")

        decoded_code = b64decode(encoded_code).decode("utf-8")
        decoded_code, is_replaced = replace_log_artifact_function(
            code=decoded_code, log_artifacts_code=log_artifacts_code
        )
        if is_replaced:
            decoded_code = (
                logger_and_consts_code + _databricks_script_code + decoded_code
            )
        else:
            decoded_code = (
                logger_and_consts_code
                + log_artifacts_code
                + _databricks_script_code
                + decoded_code
            )
        if original_handler:
            decoded_code += f"\nresult = {original_handler}(**handler_arguments)\n"
            decoded_code += _return_artifacts_code
        return b64encode(decoded_code.encode("utf-8")).decode("utf-8")

    def get_internal_parameters(self, runobj: RunObject):
        """
        Return the internal function parameters + code.
        """
        task_parameters = runobj.spec.parameters.get("task_parameters", {})
        if "original_handler" in task_parameters:
            original_handler = task_parameters["original_handler"]
        else:
            original_handler = runobj.spec.handler or ""
        log_artifacts_code, artifact_json_path = get_log_artifacts_code(
            runobj=runobj, task_parameters=task_parameters
        )
        returns = runobj.spec.returns or []
        self._verify_returns(returns=returns)
        code = self._get_modified_user_code(
            original_handler=original_handler,
            log_artifacts_code=log_artifacts_code,
        )
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
        state_thresholds: Optional[Dict[str, int]] = None,
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
            state_thresholds=state_thresholds,
            **launcher_kwargs,
        )


logger_and_consts_code = """ \n
import os
import logging
mlrun_logger = logging.getLogger('mlrun_logger')
mlrun_logger.setLevel(logging.DEBUG)

mlrun_console_handler = logging.StreamHandler()
mlrun_console_handler.setLevel(logging.DEBUG)
mlrun_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mlrun_console_handler.setFormatter(mlrun_formatter)
mlrun_logger.addHandler(mlrun_console_handler)

mlrun_default_artifact_template = 'mlrun_return_value_'
mlrun_artifact_index = 0
"""

_databricks_script_code = """

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('handler_arguments')
handler_arguments = parser.parse_args().handler_arguments
handler_arguments = json.loads(handler_arguments)


"""

log_artifacts_code_template = """\n
def mlrun_log_artifact(name='', path=''):
    global mlrun_artifact_index
    mlrun_artifact_index+=1  #  by how many artifacts we tried to log, not how many succeed.
    if name is None or name == '':
        name = f'{{mlrun_default_artifact_template}}{{mlrun_artifact_index}}'
    if not path:
        mlrun_logger.error(f'path required for logging an mlrun artifact - {{name}} : {{path}}')
        return
    if not isinstance(name, str) or not isinstance(path, str):
        mlrun_logger.error(f'name and path must be in string type for logging an mlrun artifact - {{name}} : {{path}}')
        return
    if not path.startswith('/dbfs') and not path.startswith('dbfs:/'):
        mlrun_logger.error(f'path for an mlrun artifact must start with /dbfs or dbfs:/ - {{name}} : {{path}}')
        return
    mlrun_artifacts_path = '{}'
    try:
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
        success_log = f'successfully wrote artifact details to the artifact JSON file in DBFS - {{name}} : {{path}}'
        mlrun_logger.info(success_log)
    except Exception as unknown_exception:
        mlrun_logger.error(f'log mlrun artifact failed - {{name}} : {{path}}. error: {{unknown_exception}}')
\n
"""

_return_artifacts_code = """\n
if result:
    if isinstance(result, dict):
        for key, path in result.items():
            mlrun_log_artifact(name=key, path=path)
    elif isinstance(result, (list, tuple, set)):
        for artifact_path in result:
            mlrun_log_artifact(path=artifact_path)
    elif isinstance(result, str):
        mlrun_log_artifact(path=result)
    else:
        mlrun_logger.warning(f'can not log artifacts with the result of handler function \
- result in unsupported type. {type(result)}')
"""
