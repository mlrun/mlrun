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
import json
import sys

import mlrun.utils
from mlrun.utils.singleton import Singleton

if sys.version_info >= (3, 7):
    from importlib.resources import read_text
else:
    from importlib_resources import read_text


class Version(metaclass=Singleton):
    def __init__(self):
        # When installing un-released version (e.g. by doing pip install git+https://github.com/mlrun/mlrun@development)
        # it won't have a version file, so adding some sane defaults
        self.version_info = {"git_commit": "unknown", "version": "0.0.0+unstable"}
        self.python_version = self._resolve_python_version()
        try:
            self.version_info = json.loads(
                read_text("mlrun.utils.version", "version.json")
            )
        except Exception:
            mlrun.utils.logger.warning(
                "Failed resolving version info. Ignoring and using defaults"
            )

    def _resolve_python_version(self) -> sys.version_info:
        self.python_version = sys.version_info

    def get(self):
        return self.version_info

    def get_python_version(self, as_str: bool = True):
        if as_str:
            return f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}"
        return self.python_version
