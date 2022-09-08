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
from mlrun import new_task, run_local
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestDB(TestMLRunSystem):

    project_name = "db-system-test-project"

    def test_db_commands(self):
        self._logger.debug("Creating dummy task for db queries")

        # {{run.uid}} will be substituted with the run id, so output will be written to different directories per run
        output_path = str(self.results_path / "{{run.uid}}")
        task = (
            new_task(name="demo", params={"p1": 5}, artifact_path=output_path)
            .with_secrets("file", self.assets_path / "secrets.txt")
            .set_label("type", "demo")
        )
        runs_count_before_run = len(self._run_db.list_runs(project=self.project_name))
        artifacts_count_before_run = len(
            self._run_db.list_artifacts(project=self.project_name, tag="*")
        )

        self._logger.debug("Running dummy task")
        run_object = run_local(
            task, command="training.py", workdir=str(self.assets_path)
        )
        self._logger.debug(
            "Finished running dummy task", run_object=run_object.to_dict()
        )

        self._run_uid = run_object.uid()

        runs = self._run_db.list_runs(project=self.project_name)
        assert len(runs) == runs_count_before_run + 1

        self._verify_run_metadata(
            runs[0]["metadata"],
            uid=self._run_uid,
            name="demo",
            project=self.project_name,
            labels={"kind": "", "framework": "sklearn"},
        )
        self._verify_run_spec(
            runs[0]["spec"],
            parameters={"p1": 5, "p2": "a-string"},
            inputs={"infile.txt": str(self.assets_path / "infile.txt")},
            outputs=[],
            output_path=str(self.results_path / self._run_uid),
            secret_sources=[],
            data_stores=[],
        )

        artifacts = self._run_db.list_artifacts(project=self.project_name, tag="*")
        assert len(artifacts) == artifacts_count_before_run + 4
        for artifact_key in ["chart", "html_result", "model", "mydf"]:
            artifact_exists = False
            for artifact in artifacts:
                if artifact["metadata"]["key"] == artifact_key:
                    artifact_exists = True
                    break
            assert artifact_exists

        # Verify that ArtifactList methods process result properly
        result_keys = artifacts.to_df().to_dict(orient="list")["key"]
        for artifact_key in ["chart", "html_result", "model", "mydf"]:
            assert artifact_key in result_keys
