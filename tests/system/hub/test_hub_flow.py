# Copyright 2025 Iguazio
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


import mlrun
import mlrun.common.schemas
import tests.system.base


class TestServingWithHubSteps(tests.system.base.TestMLRunSystem):
    project_name = "test-serving-with-hub-steps"
    image: str = "mlrun/mlrun"

    def test_serving_with_hub_steps(self):
        code_path = str(self.assets_path / "custom_step.py")
        project = mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )
        fn = project.set_function(
            code_path, name="serving-fn", kind="serving", image=self.image
        )
        graph = fn.set_topology("flow", engine="async")
        schema = ["id", "height", "weight"]
        graph.to(class_name="hub://verify_schema", name="verify", schema=schema).to(
            class_name="Echo", name="echo"
        ).respond()
        fn.deploy()
        serving_fn = project.get_function("serving-fn")
        event = {"id": "3425", "height": 157, "weight": 58}
        response = serving_fn.invoke("/", body=event)
        assert response == event
