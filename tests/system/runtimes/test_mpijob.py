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
import mlrun
import tests.system.base
from mlrun.runtimes.constants import RunStates


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestMpiJobRuntime(tests.system.base.TestMLRunSystem):
    project_name = "does-not-exist-mpijob"

    def test_run_state_completion(self):
        code_path = str(self.assets_path / "mpijob_function.py")

        # Create the open mpi function:
        mpijob_function = mlrun.code_to_function(
            name="mpijob_test",
            kind="mpijob",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mallonn/ml-models:1.3.0",
            requirements=["mpi4py"],
        )
        mpijob_function.spec.replicas = 4

        mpijob_run = mpijob_function.run(auto_build=True)
        assert mpijob_run.status.state == RunStates.completed

        mpijob_time = mpijob_run.status.results["time"]
        mpijob_result = mpijob_run.status.results["result"]
        assert mpijob_time is not None
        assert mpijob_result == 1000
