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

NOT_EXIST_VALUE = 'handler_not_exist'


class DatabricksRuntime(KubejobRuntime):
    kind = "databricks"
    _is_remote = True

    def get_internal_parameters(self, runobj: RunObject):
        """
        Return the internal function code.
        """
        internal_handler = runobj.spec.parameters.get("task_parameters", {}).get('internal_handler', NOT_EXIST_VALUE)
        # None and '' can be a symbol for none internal handler and don't need to be replaced.
        if internal_handler == NOT_EXIST_VALUE:
            internal_handler = runobj.spec.handler or ''
        encoded_code = (
            self.spec.build.functionSourceCode if hasattr(self.spec, "build") else None
        )
        if not encoded_code:
            print('not encoded_code')
            return ''
        decoded_code = b64decode(encoded_code).decode("utf-8")
        code = _databricks_script_code + decoded_code
        if internal_handler:
            code += f"\n{internal_handler}(**handler_arguments)\n"
        code = b64encode(code.encode("utf-8")).decode("utf-8")
        return code, internal_handler

    def _pre_run(self, runspec: RunObject, execution):
        print(f'before_pre_run: task_parameters:{runspec.spec.parameters.get("task_parameters", {})}')
        internal_code, internal_handler = self.get_internal_parameters(runspec)
        print(f"before_pre_run: internal_code raw: {internal_code}")
        if internal_code:
            task_parameters = runspec.spec.parameters.get("task_parameters", {})
            task_parameters["spark_app_code"] = internal_code
            if internal_handler:
                task_parameters["internal_handler"] = internal_handler
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
        after_task_params = runspec.spec.parameters.get("task_parameters", {})
        print(f'after_pre_run: task_parameters:{after_task_params}')
        print(f'after_pre_run: code: {b64decode(task_parameters["spark_app_code"]).decode("utf-8")}')


_databricks_script_code = """

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('handler_arguments')
handler_arguments = parser.parse_args().handler_arguments
handler_arguments = json.loads(handler_arguments)

"""


