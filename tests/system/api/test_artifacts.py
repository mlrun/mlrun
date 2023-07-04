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

import pathlib

import pytest

import mlrun.artifacts
import mlrun.common.schemas
import mlrun.errors
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAPIArtifacts(TestMLRunSystem):
    project_name = "db-system-test-project"

    @pytest.mark.enterprise
    def test_fail_overflowing_artifact(self):
        """
        Test that we fail when trying to (inline) log an artifact that is too big
        This is done to ensure that we don't corrupt the DB while truncating the data
        """
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        function = mlrun.code_to_function(
            name="test-func",
            project=self.project_name,
            filename=filename,
            handler="log_artifact_test_function",
            kind="job",
            image="mlrun/mlrun",
        )
        task = mlrun.new_task()

        # run artifact field is MEDIUMBLOB which is limited to 16MB by mysql
        # overflow and expect it to fail execution and not allow db to truncate the data
        # to avoid data corruption
        with pytest.raises(mlrun.runtimes.utils.RunError):
            function.run(
                task, params={"body_size": 16 * 1024 * 1024 + 1, "inline": True}
            )

        runs = mlrun.get_run_db().list_runs()
        assert len(runs) == 1, "run should not be created"
        run = runs[0]
        assert run["status"]["state"] == "error", "run should fail"
        assert (
            "Failed committing changes to DB" in run["status"]["error"]
        ), "run should fail with a reason"
