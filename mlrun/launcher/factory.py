# Copyright 2023 MLRun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mlrun.config
import mlrun.errors
import mlrun.launcher.base
import mlrun.launcher.local
import mlrun.launcher.remote


class LauncherFactory(object):
    @staticmethod
    def create_launcher(
        is_remote, local: bool = False
    ) -> mlrun.launcher.base.BaseLauncher:
        """
        Creates the appropriate launcher for the specified run.
        ServerSideLauncher - if running as API.
        ClientRemoteLauncher - if run is remote and local was not specified.
        ClientLocalLauncher - if run is not remote or local was specified.
        """
        if mlrun.config.is_running_as_api():
            if local:
                raise mlrun.errors.MLRunInternalServerError(
                    "Launch of local run inside the server is not allowed"
                )

            from mlrun.api.launcher import ServerSideLauncher

            return ServerSideLauncher()

        if is_remote and not local:
            return mlrun.launcher.remote.ClientRemoteLauncher()

        return mlrun.launcher.local.ClientLocalLauncher(local)
