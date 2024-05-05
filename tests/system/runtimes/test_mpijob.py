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

import mlrun
import tests.system.base
from mlrun.common.runtimes.constants import RunStates


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestMpiJobRuntime(tests.system.base.TestMLRunSystem):
    project_name = "does-not-exist-mpijob"

    def test_mpijob_run(self):
        """
        Run the `handler` function in mpijob_function.py as an OpenMPI job and validate it ran properly (see the
        docstring of the handler for more details).
        """
        code_path = str(self.assets_path / "mpijob_function.py")
        replicas = 2

        mpijob_function = mlrun.code_to_function(
            name="mpijob-test",
            kind="mpijob",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        mpijob_function.spec.replicas = replicas

        mpijob_run = mpijob_function.run(returns=["reduced_result", "rank_0_result"])
        assert mpijob_run.status.state == RunStates.completed

        reduced_result = mpijob_run.status.results["reduced_result"]
        rank_0_result = mpijob_run.status.results["rank_0_result"]
        assert reduced_result == replicas * 10
        assert rank_0_result == 1000
