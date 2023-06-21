# Copyright 2018 Iguazio
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
import pathlib
import pickle

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

    def test_model_artifact_tags(self):
        project_name = "my-test-project"
        model_name = "test_model"
        tag_1 = "tag_1"
        tag_2 = "tag_2"
        models_path = f"v3io:///projects/{project_name}/artifacts/{model_name}/"

        def _validate_artifact(expected_tags):

            # get the model from api, and check that the tags are as expected
            response = self._run_db.api_call(
                "GET",
                f"projects/{project_name}/artifacts?category=model&format=full&best-iteration=true",
            )
            assert response.status_code == 200, "failed to get artifacts"
            for artifact in response.json().get("artifacts", []):
                tag = artifact.get("metadata").get("tag")
                assert tag in expected_tags, "given tag is not in expected tags"

                # make sure artifact doesn't have the model_spec.yaml in spec.extra_data
                extra_data = artifact.get("spec", {}).get("extra_data")
                if extra_data:
                    assert (
                        "model_spec.yaml" not in extra_data
                    ), "model spec should not be in extra data"

            # get the model spec via sdk, and check that the tag is there
            _, model_obj, _ = mlrun.artifacts.get_model(models_path)
            model_dict = model_obj.to_dict()
            assert (
                "tag" not in model_dict["metadata"]
            ), "tag should not be in model metadata"

        try:
            pickle_filename = "my_model.pkl"
            with open(pickle_filename, "wb") as file:
                pickle.dump({"obj": "my-model"}, file)

            # create a project and log the model
            mlrun.set_environment(project=project_name)
            project = mlrun.get_or_create_project(project_name, context="./")

            project.set_model_monitoring_credentials(os.environ.get("V3IO_ACCESS_KEY"))
            project.log_model(
                model_name,
                body="model body",
                model_file="trained_model.pkl",
                tag=tag_1,
            )

            _validate_artifact(["latest", tag_1])

            # tag the model with another tag
            _, model_obj, _ = mlrun.artifacts.get_model(models_path)
            db = mlrun.get_run_db()
            db.tag_artifacts(
                artifacts=[model_obj],
                project=project_name,
                tag_name=tag_2,
                replace=False,
            )

            _validate_artifact(["latest", tag_1, tag_2])

        finally:
            self._delete_test_project(project_name)
