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
import os
from pathlib import Path

import pytest
import yaml

import mlrun
import tests.system.base
from mlrun.runtimes.function_reference import FunctionReference
from tests.datastore.databricks_utils import is_databricks_configured

here = Path(__file__).absolute().parent
config_file_path = here / "assets" / "test_databricks.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)


@pytest.mark.skipif(
    not is_databricks_configured(config_file_path),
    reason="databricks parameters not configured",
)
@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestDatabricksRuntime(tests.system.base.TestMLRunSystem):
    project_name = "databricks-system-test"

    def setup_class(self):
        for key, value in config["env"].items():
            if value is not None:
                os.environ[key] = value

    @staticmethod
    def _add_databricks_env(function, is_cluster_id_required):
        cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID", None)
        if not cluster_id and is_cluster_id_required:
            raise KeyError(
                "The environment variable 'DATABRICKS_CLUSTER_ID' is not set, and it is required for this test."
            )
        project = mlrun.get_or_create_project(
            "databricks-proj", context="./", user_project=False
        )

        job_env = {
            "DATABRICKS_HOST": os.environ["DATABRICKS_HOST"],
        }
        if is_cluster_id_required:
            job_env["DATABRICKS_CLUSTER_ID"] = cluster_id

        secrets = {"DATABRICKS_TOKEN": os.environ["DATABRICKS_TOKEN"]}

        project.set_secrets(secrets)

        for name, val in job_env.items():
            function.spec.env.append({"name": name, "value": val})

    def test_kwargs_from_code(self):

        code = """

def print_kwargs(**kwargs):
    print(f"kwargs: {kwargs}")
        """

        function_ref = FunctionReference(
            kind="databricks",
            code=code,
            image="mlrun/mlrun",
            name="databricks-test",
        )

        function = function_ref.to_function()

        self._add_databricks_env(function=function, is_cluster_id_required=True)

        run = function.run(
            handler="print_kwargs",
            project="databricks-proj",
            params={
                "mlrun_internal_timeout_minutes": 15,
                "param1": "value1",
                "param2": "value2",
            },
        )
        assert run.status.state == "completed"
        assert (
            run.status.results["databricks_runtime_task"]["logs"]
            == "kwargs: {'param1': 'value1', 'param2': 'value2'}\n"
        )

    def test_kwargs_from_file(self):
        code_path = str(self.assets_path / "databricks_function_print_kwargs.py")
        function = mlrun.code_to_function(
            name="function-with-args",
            kind="databricks",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )

        self._add_databricks_env(function=function, is_cluster_id_required=True)

        run = function.run(
            handler="func",
            auto_build=True,
            project="databricks-proj",
            params={
                "mlrun_internal_timeout_minutes": 15,
                "param1": "value1",
                "param2": "value2",
            },
        )
        assert run.status.state == "completed"
        assert (
            run.status.results["databricks_runtime_task"]["logs"]
            == "{'param1': 'value1', 'param2': 'value2'}\n"
        )
