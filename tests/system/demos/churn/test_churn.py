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
import pytest

import mlrun
from tests.system.base import TestMLRunSystem
from tests.system.demos.base import TestDemo


# Marked as enterprise because of v3io mount and pipelines
@pytest.mark.skip("not up to date demos needs to run demos from mlrun/demos repo")
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
            image="mlrun/mlrun-gpu" if self.use_gpus else "mlrun/mlrun",
            description="clean and encode raw data",
            categories=["data-prep"],
            labels={"author": "yasha", "framework": "xgboost"},
        ).apply(mlrun.mount_v3io())

        clean_data_function.spec.remote = True
        clean_data_function.spec.replicas = 1
        clean_data_function.spec.service_type = "NodePort"

        self._logger.debug("Setting project functions")
        demo_project.set_function(clean_data_function)
        demo_project.set_function("hub://describe", "describe")
        demo_project.set_function("hub://xgb-trainer", "classify")
        demo_project.set_function("hub://xgb-test", "xgbtest")
        demo_project.set_function("hub://coxph-trainer", "survive")
        demo_project.set_function("hub://coxph-test", "coxtest")
        demo_project.set_function("hub://churn-server", "server")

        self._logger.debug("Setting project workflow")
        demo_project.set_workflow(
            "main", str(self.assets_path / "workflow.py"), embed=True
        )

        return demo_project

    def test_demo(self):
        self.run_and_verify_project(runs_amount=6)
