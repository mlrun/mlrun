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
from mlrun.launcher.base import _BaseLauncher
from mlrun.launcher.local import LocalLauncher
from mlrun.launcher.remote import RemoteLauncher


class LauncherFactory(object):
    @staticmethod
    def create_client_side_launcher(local) -> _BaseLauncher:
        """create LocalLauncher or RemoteLauncher according to the if local run was specified"""
        if local:
            return LocalLauncher()
        return RemoteLauncher()
