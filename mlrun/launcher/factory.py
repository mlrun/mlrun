# Copyright 2023 Iguazio
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
from dependency_injector import containers, providers

import mlrun.config
import mlrun.errors
import mlrun.launcher.base
import mlrun.launcher.local
import mlrun.launcher.remote
import mlrun.utils.singleton


class LauncherFactory(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self):
        self._launcher_container = LauncherContainer()

    def create_launcher(
        self, is_remote: bool, **kwargs
    ) -> mlrun.launcher.base.BaseLauncher:
        """
        Creates the appropriate launcher for the specified run.
        ServerSideLauncher - if running as API.
        ClientRemoteLauncher - if the run is remote and local was not specified.
        ClientLocalLauncher - if the run is not remote or local was specified.

        :param is_remote:   Whether the runtime requires remote execution.

        :return:            The appropriate launcher for the specified run.
        """
        if mlrun.config.is_running_as_api():
            return self._launcher_container.server_side_launcher(**kwargs)

        local = kwargs.get("local", False)
        if is_remote and not local:
            return self._launcher_container.client_remote_launcher(**kwargs)

        return self._launcher_container.client_local_launcher(**kwargs)


class LauncherContainer(containers.DeclarativeContainer):
    client_remote_launcher = providers.Factory(
        mlrun.launcher.remote.ClientRemoteLauncher
    )
    client_local_launcher = providers.Factory(mlrun.launcher.local.ClientLocalLauncher)

    # Provider for injection of a server side launcher.
    # This allows us to override the launcher from external packages without having to import them.
    server_side_launcher = providers.Factory(mlrun.launcher.base.BaseLauncher)
