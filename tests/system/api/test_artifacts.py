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
import tempfile
import uuid

import pytest

from tests.system.base import TestMLRunSystem

function_path = str(pathlib.Path(__file__).parent / "assets" / "function.py")


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAPIArtifacts(TestMLRunSystem):
    project_name = "db-system-test-project"

    @pytest.mark.enterprise
    def test_import_artifact(self):
        temp_dir = tempfile.mkdtemp()
        key = f"artifact_key_{uuid.uuid4()}"
        body = "my test artifact"
        artifact = self.project.log_artifact(
            key, body=body, local_path=f"{temp_dir}/test_artifact.txt"
        )
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".yaml", delete=True
        ) as temp_file:
            artifact.export(temp_file.name)
            artifact = self.project.import_artifact(
                temp_file.name, new_key=f"imported_artifact_key_{uuid.uuid4()}"
            )
        assert artifact.to_dataitem().get().decode() == body

    @pytest.mark.enterprise
    def test_verify_artifact_tag_in_output(self):
        # log the same artifact but with different tags and check the function output
        self.project.set_function(
            func=function_path,
            handler="log_artifact_many_tags",
            name="test",
            image="mlrun/mlrun",
        )
        run = self.project.run_function("test")
        output_uri = run.output("file_result")
        outputs_uri = run.outputs["file_result"]

        assert "v3" in output_uri, "Expected 'v3' tag in output_uri"
        assert "v3" in outputs_uri, "Expected 'v3' tag in outputs_uri"

        runs = self.project.list_runs()
        first_run = runs.to_objects()[0]
        output_uri_from_list_runs = first_run.output("file_result")
        outputs_uri_from_list_runs = first_run.outputs["file_result"]
        assert (
            "v3" in output_uri_from_list_runs
        ), "Expected 'v3' tag in output_uri_from_list_runs"
        assert (
            "v3" in outputs_uri_from_list_runs
        ), "Expected 'v3' tag in outputs_uri_from_list_runs"

        # run the same function with different tags
        self.project.set_function(
            func=function_path,
            handler="log_artifact_with_tag",
            name="test",
            image="mlrun/mlrun",
        )

        func_v1_run = self.project.run_function("test", params={"tag": "v1"})
        output_uri = func_v1_run.output("file_result")
        outputs_uri = func_v1_run.outputs["file_result"]

        assert "v1" in output_uri, "Expected 'v1' tag in output_uri"
        assert "v1" in outputs_uri, "Expected 'v1' tag in outputs_uri"
