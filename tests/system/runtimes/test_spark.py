# Copyright 2026 Iguazio
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
import pytest

import mlrun
import mlrun.common.runtimes.constants
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestSpark(tests.system.base.TestMLRunSystem):
    project_name = "spark-system-test"

    @pytest.mark.collect_pod_logs
    def test_run_on_deployed_default_image(self):
        if not (mlrun.mlconf.spark_app_image and mlrun.mlconf.spark_app_image_tag):
            pytest.skip("Spark application image is not configured")

        mlrun.runtimes.Spark3Runtime.deploy_default_image()

        function = mlrun.code_to_function(
            name="spark-default-image",
            kind="spark",
            project=self.project_name,
            filename=str(self.assets_path / "spark_session.py"),
            image=mlrun.runtimes.Spark3Runtime.default_mlrun_image,
        )
        function.with_driver_requests(cpu=1, mem="512m")
        function.with_executor_requests(cpu=1, mem="512m")

        run = function.run()

        assert run.status.state == mlrun.common.runtimes.constants.RunStates.completed
        assert run.outputs["spark_result"] == 1000
