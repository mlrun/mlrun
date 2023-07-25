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
#
import datetime
import os
from sys import executable
from pathlib import Path
import pytest
import yaml

import mlrun
from mlrun.runtimes.function_reference import FunctionReference
import tests.system.base

here = Path(__file__).absolute().parent
config_file_path = here.parent / "assets" / "test-databricks_runtime.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)


def exec_run(args):
    cmd = [executable, "-m", "mlrun", "run"] + args
    out = os.popen(" ".join(cmd)).read()
    return out


MUST_HAVE_VARIABLES = ["DATABRICKS_TOKEN", "DATABRICKS_HOST"]


def is_databricks_env_configured():
    env_params = config["env"]
    for necessary_variable in MUST_HAVE_VARIABLES:
        if env_params.get(necessary_variable, None) is None:
            return False
    return True


@pytest.mark.skipif(
    not is_databricks_env_configured(),
    reason="databricks parameters not configured",
)
@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestDatabricksRuntime(tests.system.base.TestMLRunSystem):
    mandatory_enterprise_env_vars = tests.system.base.TestMLRunSystem.mandatory_enterprise_env_vars + [
        "DATABRICKS_TOKEN", "DATABRICKS_HOST"]
    project_name = "databricks-system-test"

    # def test_test(self):
    #     print(os.environ)
    #     print()

    def test_with_function_reference(self):
        cluster_id = os.environ["DATABRICKS_CLUSTER_ID"]
        if not cluster_id:
            raise KeyError(
                f"The environment variable 'DATABRICKS_CLUSTER_ID' is not set, and it is required for this test.")
        code = """

def print_args(**kwargs):
    print(f"kwargs: {kwargs}")
        """
        # **Databricks cluster credentials**

        project = mlrun.get_or_create_project("databricks-proj", context="./", user_project=False)

        job_env = {
            "DATABRICKS_HOST": os.environ["DATABRICKS_HOST"],
            "DATABRICKS_CLUSTER_ID": os.environ["DATABRICKS_CLUSTER_ID"]
        }

        secrets = {
            "DATABRICKS_TOKEN": os.environ["DATABRICKS_TOKEN"]
        }

        project.set_secrets(secrets)

        # **Define and run the function**

        function_ref = FunctionReference(
            kind="databricks-job",
            code=code,
            image="tomermamia855/mlrun-api:tomer-databricks-runtime", #  TODO replace it after PR
            name="databricks-test",
        )

        function = function_ref.to_function()

        for name, val in job_env.items():
            function.spec.env.append({
                'name': name,
                'value': val
            })

        function.run(
            handler="print_args",
            auto_build=True,
            params={"param1": "value1", "param2": "value2"}
        )