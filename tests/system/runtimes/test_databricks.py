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
import copy
import os
import time
import uuid
from datetime import datetime, timedelta

import pandas as pd
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

import mlrun
import tests.system.base
from mlrun.errors import MLRunRuntimeError
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.runtimes.utils import RunError
from tests.datastore.databricks_utils import MLRUN_ROOT_DIR, teardown_dbfs_dirs

print_kwargs_function = """

def %s(**kwargs):
    print(f"kwargs: {kwargs}")
"""

default_test_params = {
    "task_parameters": {"timeout_minutes": 15},
    "param1": "value1",
    "param2": "value2",
}


class TestDatabricksSystem(tests.system.base.TestMLRunSystem):
    mandatory_env_vars = [
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
        "DATABRICKS_CLUSTER_ID",
    ] + tests.system.base.TestMLRunSystem.mandatory_env_vars


@TestDatabricksSystem.skip_test_if_env_not_configured
class TestDatabricksRuntime(tests.system.base.TestMLRunSystem):
    project_name = "databricks-system-test"

    @staticmethod
    def _get_active_run_by_name(workspace: WorkspaceClient, run_name: str):
        previous_day_utc_time = datetime.utcnow() - timedelta(days=1)
        previous_day_utc_time_in_ms = int(previous_day_utc_time.timestamp() * 1000)
        # in order to optimize the query, list_runs is filtered by time and active_only.
        runs = list(
            workspace.jobs.list_runs(
                active_only=True, start_time_from=previous_day_utc_time_in_ms
            )
        )
        # We use startswith because we append a timestamp at the end of the run name.
        runs_by_run_name = [
            databricks_run
            for databricks_run in runs
            if databricks_run.run_name.startswith(run_name)
        ]
        if len(runs_by_run_name) == 0:
            raise MLRunRuntimeError(
                f"No active runs were found in Databricks with run_name={run_name}"
            )
        elif len(runs_by_run_name) > 1:
            raise MLRunRuntimeError(
                f"Too many active runs were found in Databricks with run_name={run_name}"
            )
        return runs[0]

    def _abort_run(self):
        self._logger.info("start aborting")
        mlrun_runs = self.project.list_runs(state="running")
        if len(mlrun_runs) < 1:
            raise MLRunRuntimeError(
                f"No active runs related to project {self.project_name} could be found"
            )
        if len(mlrun_runs) > 1:
            raise MLRunRuntimeError(
                f"Too many active runs related to project {self.project_name} were found"
            )
        mlrun_run = mlrun_runs.to_objects()[0]
        self._run_db.abort_run(uid=mlrun_run.uid(), project=self.project_name)

    def _check_artifacts_by_path(self, paths_dict):
        artifacts = self.project.list_artifacts().to_objects()
        assert len(artifacts) == len(paths_dict)
        #  TODO change to get_artifact by key in the future if it will be user-defined.
        for local_path, expected_dbfs_path in paths_dict.items():
            artifacts_by_path = [
                artifact
                for artifact in artifacts
                if artifact.spec.src_path == f"dbfs://{expected_dbfs_path}"
            ]
            assert len(artifacts_by_path) == 1
            artifact = artifacts_by_path[0]
            artifact_df = artifact.to_dataitem().as_df()
            if local_path.endswith(".parquet"):
                expected_df = pd.read_parquet(local_path)
            elif local_path.endswith(".csv"):
                expected_df = pd.read_csv(local_path)
            else:
                raise ValueError(
                    "The test does not support files that are not in the Parquet or CSV format."
                )
            pd.testing.assert_frame_equal(expected_df, artifact_df)

    def setup_method(self, method):
        time.sleep(2)  # For project handling...
        super().setup_method(method)

    def setup_class(self):
        super().setup_class()
        self.test_folder_name = "/databricks_system_test"
        self.dbfs_folder_path = f"{MLRUN_ROOT_DIR}{self.test_folder_name}"
        self.workspace = WorkspaceClient()

    def teardown_class(self):
        teardown_dbfs_dirs(
            workspace=self.workspace, specific_test_class_dir=self.test_folder_name
        )

    @classmethod
    def assert_print_kwargs(cls, print_kwargs_run):
        assert print_kwargs_run.status.state == "completed"
        logs = cls._run_db.get_log(uid=print_kwargs_run.uid())[1].decode()
        assert "{'param1': 'value1', 'param2': 'value2'}\n" in logs

    def _add_databricks_env(self, function, is_cluster_id_required=True):
        cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID", None)
        if not cluster_id and is_cluster_id_required:
            raise KeyError(
                "The environment variable 'DATABRICKS_CLUSTER_ID' is not set, and it is required for this test."
            )

        job_env = {
            "DATABRICKS_HOST": os.environ["DATABRICKS_HOST"],
        }
        if is_cluster_id_required:
            job_env["DATABRICKS_CLUSTER_ID"] = cluster_id

        secrets = {"DATABRICKS_TOKEN": os.environ["DATABRICKS_TOKEN"]}

        self.project.set_secrets(secrets)

        for name, val in job_env.items():
            function.spec.env.append({"name": name, "value": val})

    @pytest.mark.parametrize(
        "use_existing_cluster, fail", [(True, False), (False, True), (False, False)]
    )
    def test_kwargs_from_code(self, use_existing_cluster, fail):
        code = print_kwargs_function % "print_kwargs"
        function_ref = FunctionReference(
            kind="databricks",
            code=code,
            image="mlrun/mlrun",
            name="databricks-test",
        )

        function = function_ref.to_function()

        self._add_databricks_env(
            function=function, is_cluster_id_required=use_existing_cluster
        )
        params = copy.deepcopy(default_test_params)
        if fail:
            params["task_parameters"]["new_cluster_spec"] = {
                "node_type_id": "this is not a real node type so it should fail"
            }
            with pytest.raises(mlrun.runtimes.utils.RunError):
                run = function.run(
                    handler="print_kwargs",
                    project=self.project_name,
                    params=params,
                )
                assert run.status.state == "error"
        else:
            run = function.run(
                handler="print_kwargs",
                project=self.project_name,
                params=params,
            )
            self.assert_print_kwargs(print_kwargs_run=run)

    def test_failure_in_databricks(self):
        code = """

def import_mlrun():
    import mlrun
"""

        function_ref = FunctionReference(
            kind="databricks",
            code=code,
            image="mlrun/mlrun",
            name="databricks-fails-test",
        )

        function = function_ref.to_function()

        self._add_databricks_env(function=function)
        with pytest.raises(RunError) as error:
            function.run(
                handler="import_mlrun",
                project=self.project_name,
            )
        lines = str(error.value).splitlines()
        assert "No module named 'mlrun'" in lines[2]
        assert "No module named 'mlrun'" in lines[-1]

    def test_kwargs_from_file(self):
        code_path = str(self.assets_path / "databricks_function_print_kwargs.py")
        function = mlrun.code_to_function(
            name="function-with-args",
            kind="databricks",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )

        self._add_databricks_env(function=function)

        run = function.run(
            handler="func",
            auto_build=True,
            project=self.project_name,
            params=default_test_params,
        )
        assert run.status.state == "completed"
        self.assert_print_kwargs(print_kwargs_run=run)

    @pytest.mark.parametrize(
        "handler, function_name",
        [
            ("print_kwargs", "print_kwargs"),
            (
                "",
                "handler",
            ),  # test default handler.
        ],
    )
    def test_rerun(self, handler, function_name):
        databricks_code = print_kwargs_function % function_name
        function_kwargs = {"handler": handler} if handler else {}
        function_ref = FunctionReference(
            kind="databricks",
            code=databricks_code,
            image="mlrun/mlrun",
            name="databricks-test",
        )

        function = function_ref.to_function()

        self._add_databricks_env(function=function)
        run = function.run(
            project=self.project_name, params=default_test_params, **function_kwargs
        )
        self.assert_print_kwargs(print_kwargs_run=run)
        second_run = function.run(runspec=run, project=self.project_name)
        self.assert_print_kwargs(print_kwargs_run=second_run)

    def test_missing_code_run(self):
        function_ref = FunctionReference(
            kind="databricks",
            code="",
            image="mlrun/mlrun",
            name="databricks-test",
        )

        function = function_ref.to_function()

        self._add_databricks_env(function=function)
        with pytest.raises(mlrun.errors.MLRunBadRequestError) as bad_request_error:
            run = function.run(
                project=self.project_name,
                params=default_test_params,
                handler="not_exist_handler",
            )
            assert run.status.state == "error"
            assert (
                "Databricks function must be provided with user code"
                in bad_request_error.value
            )

    def test_abort_task(self):
        #  clean up any active runs
        if self.project.list_runs(state="running"):
            self.project = mlrun.projects.new_project(self.project_name, overwrite=True)
        sleep_code = """

import time

def handler(**kwargs):
    time.sleep(1000) # very long sleep

"""
        function_ref = FunctionReference(
            kind="databricks",
            code=sleep_code,
            image="mlrun/mlrun",
            name="databricks_abort_test",
        )
        function = function_ref.to_function()
        self._add_databricks_env(function=function)
        databricks_run_name = f"databricks_abort_test_{uuid.uuid4()}"
        params = {"task_parameters": {"databricks_run_name": databricks_run_name}}
        function.run(project=self.project_name, params=params, watch=False)
        # wait for databricks to run the function.
        time.sleep(10)
        workspace = WorkspaceClient()
        run = self._get_active_run_by_name(
            workspace=workspace, run_name=databricks_run_name
        )
        assert run.state.life_cycle_state in (
            RunLifeCycleState.PENDING,
            RunLifeCycleState.RUNNING,
        )
        self._abort_run()
        time.sleep(5)
        run = workspace.jobs.get_run(run_id=run.run_id)
        # wait for databricks to update the status.
        assert run.state.life_cycle_state in (
            RunLifeCycleState.TERMINATING,
            RunLifeCycleState.TERMINATED,
        )
        assert run.state.result_state == RunResultState.CANCELED

    def test_log_artifact(self):
        self._run_db.del_artifacts(project=self.project_name)
        artifact_key_parquet = f"my_artifact_test_{uuid.uuid4()}.parquet"
        artifact_key_csv = f"my_artifact_test_{uuid.uuid4()}.csv"
        parquet_artifact_dbfs_path = (
            f"{self.dbfs_folder_path}/test_log_artifact/{artifact_key_parquet}"
        )
        csv_artifact_dbfs_path = (
            f"{self.dbfs_folder_path}/test_log_artifact/{artifact_key_csv}"
        )
        src_parquet_path = str(self.assets_path / "test_data.parquet")
        src_csv_path = str(self.assets_path / "test_data.csv")
        paths_dict = {
            src_parquet_path: parquet_artifact_dbfs_path,
            src_csv_path: csv_artifact_dbfs_path,
        }
        with open(str(self.assets_path / "test_data.parquet"), "rb") as parquet_file:
            self.workspace.dbfs.upload(
                src=parquet_file, path=parquet_artifact_dbfs_path
            )
        with open(str(self.assets_path / "test_data.csv"), "rb") as csv_file:
            self.workspace.dbfs.upload(src=csv_file, path=csv_artifact_dbfs_path)
        #  CSV has been tested as a Spark path, and an illegal path was
        #  used for testing to avoid triggering an error in log_artifact.
        code = f"""\n
def main():
    mlrun_log_artifact('my_test_artifact_parquet','/dbfs{parquet_artifact_dbfs_path}')
    mlrun_log_artifact('illegal artifact',10)
    return {{'my_test_artifact_csv': 'dbfs:{csv_artifact_dbfs_path}'}}
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
            handler="main",
            project=self.project_name,
        )
        time.sleep(2)
        self._check_artifacts_by_path(paths_dict=paths_dict)
        self._run_db.del_artifacts(project=self.project_name)
        assert (
            len(self.project.list_artifacts()) == 0
        )  # Make sure all artifacts have been deleted.
        function.run(runspec=run, project=self.project_name)  # test rerun.
        time.sleep(3)
        self._check_artifacts_by_path(paths_dict=paths_dict)
