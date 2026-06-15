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


import pathlib

import mlrun
import mlrun.common.schemas
import tests.integration.sdk_api.base


class TestServingWithHubSteps(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_serving_with_hub_steps(self):
        function_path = str(pathlib.Path(__file__).parent / "assets" / "custom_step.py")
        project = mlrun.new_project("test-serving-with-hub-steps", save=False)
        fn = project.set_function(function_path, name="serving-fn", kind="serving")
        graph = fn.set_topology("flow", engine="async")
        schema = ["id", "height", "weight"]
        graph.to(class_name="hub://verify_schema", name="verify", schema=schema).to(
            class_name="Echo", name="echo"
        ).respond()
        server = fn.to_mock_server()
        event = {"id": "3425", "height": 157, "weight": 58}
        try:
            response = server.test("/", event)
            assert response == event
        finally:
            server.wait_for_completion()
