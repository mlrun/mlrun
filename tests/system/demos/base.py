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

import mlrun
from tests.system.base import TestMLRunSystem


class TestDemo(TestMLRunSystem):
    def custom_setup(self):

        # specifically for each workflow, this combines the artifact path above with a
        # unique path made from the workflow uid.
        self._workflow_artifact_path = os.path.join(
            mlrun.mlconf.artifact_path, "pipeline/{{workflow.uid}}"
        )
        self._demo_project = self.create_demo_project()

        self._logger.debug(
            "Created demo project",
            project_name=self.project_name,
            project=self._demo_project.to_dict(),
        )

    def create_demo_project(self) -> mlrun.projects.MlrunProject:
        raise NotImplementedError

    def run_and_verify_project(self, runs_amount: int = 1, arguments: dict = None):
        arguments = arguments or {}
        run_id = self._demo_project.run(
            "main",
            arguments=arguments,
            artifact_path=self._workflow_artifact_path,
            dirty=True,
            watch=True,
        )

        runs = self._run_db.list_runs(
            project=self.project_name, labels=f"workflow={run_id}"
        )

        self._logger.debug("Completed Runs", runs=runs)

        assert len(runs) == runs_amount
        for run in runs:
            assert run["status"]["state"] == "completed"
