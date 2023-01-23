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
import time

import mlrun
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestLogCollector(tests.system.base.TestMLRunSystem):
    custom_project_names_to_delete = []

    def custom_teardown(self):
        for name in self.custom_project_names_to_delete:
            self._delete_test_project(name)

    def test_log_collector(self):
        self._logger.debug("Testing log collector")

        code_path = str(self.assets_path / "function_with_logs.py")

        # we set the function lifecycle, which is equivalent to the number of log lines the function will print
        life_cycle_seconds = 100

        project_name = "test-log-collector"
        self.custom_project_names_to_delete.append(project_name)
        proj = mlrun.get_or_create_project(project_name, self.assets_path)
        function = mlrun.code_to_function(
            name="function-with-logs",
            kind="job",
            handler="handler",
            project=proj.name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        run = function.run(params={"life_cycle_seconds": life_cycle_seconds})

        # we retry getting logs in case the log collector hadn't started collecting logs for the function.
        # since log collecting starts in a periodic task, it might take a few seconds for it to start
        state, logs = mlrun.utils.retry_until_successful(
            3,
            12,
            self._logger,
            True,
            mlrun.get_run_db().get_log,
            uid=run.metadata.uid,
            project=proj.name,
        )

        # verify run state is not unknown
        assert state != "unknown", f"Unexpected state {state}"

        # verify the logs are not empty
        assert logs, "Expected logs to be not empty"

        self._logger.debug("Finished log collector test")
