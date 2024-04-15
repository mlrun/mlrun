# Copyright 2024 Iguazio
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
import os

import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestApplicationRuntime(tests.system.base.TestMLRunSystem):
    project_name = "application-system-test"

    def custom_setup(self):
        super().custom_setup()
        self._vizro_app_code_filename = "vizro_app.py"
        self._files_to_upload = [self._vizro_app_code_filename]

    def test_deploy_application(self):
        self._upload_code_to_cluster()

        self._logger.debug("Creating application")
        function = self.project.set_function(
            name="vizro-app",
            kind="application",
            requirements=["vizro", "gunicorn", "Werkzeug==2.2.2"],
        )
        function.set_internal_application_port(8050)
        function.spec.command = "gunicorn"
        function.spec.args = [
            "vizro_app:server",
            "--bind",
            "0.0.0.0:8050",
            "--log-level",
            "debug",
        ]

        function.with_source_archive(
            os.path.join(self.remote_code_dir, self._vizro_app_code_filename),
            pull_at_runtime=False,
        )

        self._logger.debug("Deploying vizro application")
        function.deploy(with_mlrun=False)

        assert function.invoke("/")
