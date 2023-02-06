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
import pytest

import mlrun
from tests.system.base import TestMLRunSystem
from tests.system.demos.base import TestDemo


# Marked as enterprise because of v3io mount and pipelines
@pytest.mark.skip("not up to date demos needs to run demos from mlrun/demos repo")
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestSKLearn(TestDemo):

    project_name = "sklearn-project"

    def create_demo_project(self) -> mlrun.projects.MlrunProject:
        self._logger.debug("Creating sklearn project")
        demo_project = mlrun.get_or_create_project(
            self.project_name, str(self.assets_path), init_git=True
        )

        self._logger.debug("Creating iris-generator function")
        function_path = str(self.assets_path / "iris_generator_function.py")
        iris_generator_function = mlrun.code_to_function(
            name="gen-iris",
            kind="job",
            filename=function_path,
            image="mlrun/mlrun",
        )

        iris_generator_function.spec.remote = True
        iris_generator_function.spec.replicas = 1
        iris_generator_function.spec.service_type = "NodePort"
        iris_generator_function.spec.build.commands.append(
            "pip install pandas sklearn pyarrow"
        )

        self._logger.debug("Setting project functions")
        demo_project.set_function(iris_generator_function)
        demo_project.set_function("hub://describe", "describe")
        demo_project.set_function("hub://auto_trainer", "auto_trainer")
        demo_project.set_function("hub://model_server", "serving")
        demo_project.set_function("hub://model_server_tester", "live_tester")

        self._logger.debug("Setting project workflow")
        demo_project.set_workflow(
            "main", str(self.assets_path / "workflow.py"), embed=True
        )

        return demo_project

    def test_demo(self):
        self.run_and_verify_project(runs_amount=5)
