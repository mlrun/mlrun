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
        demo_project = mlrun.new_project(
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
        demo_project.set_function("hub://sklearn_classifier", "train")
        demo_project.set_function("hub://test_classifier", "test")
        demo_project.set_function("hub://model_server", "serving")
        demo_project.set_function("hub://model_server_tester", "live_tester")

        self._logger.debug("Setting project workflow")
        demo_project.set_workflow(
            "main", str(self.assets_path / "workflow.py"), embed=True
        )

        return demo_project

    def test_demo(self):
        self.run_and_verify_project(runs_amount=5)
