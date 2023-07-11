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
from typing import Optional, Type

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
        self._launcher_cls: Optional[Type[mlrun.launcher.base.BaseLauncher]] = None

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
        if self._launcher_cls:
            return self._launcher_cls(**kwargs)

        local = kwargs.get("local", False)
        if is_remote and not local:
            return mlrun.launcher.remote.ClientRemoteLauncher(**kwargs)

        return mlrun.launcher.local.ClientLocalLauncher(**kwargs)

    def set_launcher(self, launcher_cls: Type[mlrun.launcher.base.BaseLauncher]):
        """
        Launcher setter for injection of a custom launcher.
        This allows us to override the launcher from external packages without having to import them.
        :param launcher_cls:    The launcher class to use.
        """
        self._launcher_cls = launcher_cls
