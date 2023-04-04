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
import mlrun.errors
from mlrun.launcher import ClientLocalLauncher, ClientRemoteLauncher, _BaseLauncher


class LauncherFactory(object):
    @staticmethod
    def create_launcher(local: bool = False) -> _BaseLauncher:
        """
        Creates a ServerSideLauncher if running as API.
        Otherwise, ClientLocalLauncher or ClientRemoteLauncher according to the if local run was specified.
        """
        if mlrun.mlconf.is_running_as_api:
            if local:
                raise mlrun.errors.MLRunInternalServerError(
                    "Launch of local run inside the server is not allowed"
                )

            from mlrun.api import ServerSideLauncher

            return ServerSideLauncher()

        if local:
            return ClientLocalLauncher()

        return ClientRemoteLauncher()
