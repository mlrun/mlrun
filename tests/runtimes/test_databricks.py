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
import ast
import hashlib
from base64 import b64decode, b64encode

import pytest

import mlrun.errors
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.model import RunObject
from mlrun.runtimes.databricks_job.databricks_runtime import DatabricksRuntime

USER_CODE = """ \n
def handler(**kwargs):
    print(f"kwargs: {kwargs}")

"""
USER_CODE_WITH_DUMMY = """ \n
def handler(**kwargs):
    print(f"kwargs: {kwargs}")

def mlrun_log_artifact(**kwargs):
    pass

"""


def test_prevent_run_locally():
    databricks_runtime = DatabricksRuntime()
    with pytest.raises(
        MLRunInvalidArgumentError, match="Databricks runtime cannot run locally."
    ):
        databricks_runtime.run(local=True)


@pytest.mark.parametrize("user_code", [USER_CODE, USER_CODE_WITH_DUMMY])
def test_get_internal_parameters(user_code):
    databricks_runtime = DatabricksRuntime()
    runobj = RunObject()
    runobj.metadata.uid = "1"
    runobj.spec.handler = "handler"
    encoded_user_code = b64encode(user_code.encode("utf-8")).decode("utf-8")
    databricks_runtime.spec.build.functionSourceCode = encoded_user_code
    (
        encoded_formatted_code,
        updated_task_parameters,
    ) = databricks_runtime.get_internal_parameters(runobj=runobj)

    #  test code:
    code = b64decode(encoded_formatted_code).decode("utf-8")
    parsed_code = ast.parse(code)
    expected_hash = "a67e065523171f407006611dbbcbb2a94b246af0bfb73e262a738e418354dc8a"
    already_found = False

    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef) and node.name == "mlrun_log_artifact":
            if already_found:
                raise RuntimeError("found more than mlrun_log_artifact function.")
            assert (
                hashlib.sha256(str(ast.unparse(node)).encode("utf-8")).hexdigest()
                == expected_hash
            )
            already_found = True
    assert already_found

    #  test task parameters:
    artifact_dir = mlrun.mlconf.function.databricks.artifact_directory_path
    assert updated_task_parameters == {
        "artifact_json_path": f"{artifact_dir}/mlrun_artifact_1.json",
        "original_handler": "handler",
    }
