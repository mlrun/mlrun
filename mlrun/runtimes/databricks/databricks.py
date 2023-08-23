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

from mlrun.model import RunObject
from mlrun.runtimes.kubejob import KubejobRuntime


class DatabricksRuntime(KubejobRuntime):
    kind = "databricks"
    _is_remote = True

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


_databricks_script_code = """

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('handler_arguments')
handler_arguments = parser.parse_args().handler_arguments
handler_arguments = json.loads(handler_arguments)

"""
