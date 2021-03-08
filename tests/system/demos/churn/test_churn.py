import pytest

import mlrun
from tests.system.base import TestMLRunSystem
from tests.system.demos.base import TestDemo


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestChurn(TestDemo):

    project_name = "churn-project"
    use_gpus = False

    def create_demo_project(self) -> mlrun.projects.MlrunProject:
        self._logger.debug("Creating churn project")
        demo_project = mlrun.new_project(
            self.project_name, str(self.assets_path), init_git=True
        )

        data_url = (
            "https://raw.githubusercontent.com/mlrun/demos/master/customer-churn-prediction/WA_Fn-UseC_-Telco-"
            "Customer-Churn.csv"
        )
        demo_project.log_artifact("raw-data", target_path=data_url)

        self._logger.debug("Creating clean-data function")
        function_path = str(self.assets_path / "data_clean_function.py")
        clean_data_function = mlrun.code_to_function(
            name="clean_data",
            kind="job",
            filename=function_path,
            image="mlrun/ml-models-gpu" if self.use_gpus else "mlrun/ml-models",
            description="clean and encode raw data",
            categories=["data-prep"],
            labels={"author": "yasha", "framework": "xgboost"},
        ).apply(mlrun.mount_v3io())

        clean_data_function.spec.remote = True
        clean_data_function.spec.replicas = 1
        clean_data_function.spec.service_type = "NodePort"
        clean_data_function.spec.command = function_path

        self._logger.debug("Setting project functions")
        demo_project.set_function(clean_data_function)
        demo_project.set_function("hub://describe", "describe")
        demo_project.set_function("hub://xgb_trainer", "classify")
        demo_project.set_function("hub://xgb_test", "xgbtest")
        demo_project.set_function("hub://coxph_trainer", "survive")
        demo_project.set_function("hub://coxph_test", "coxtest")
        demo_project.set_function("hub://churn_server", "server")

        self._logger.debug("Setting project workflow")
        demo_project.set_workflow(
            "main", str(self.assets_path / "workflow.py"), embed=True
        )

        return demo_project

    def test_demo(self):
        self.run_and_verify_project(runs_amount=6)
