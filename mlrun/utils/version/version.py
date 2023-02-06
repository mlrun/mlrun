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


class _VersionInfo:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"


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

    def get(self):
        return self.version_info

    def get_python_version(self) -> _VersionInfo:
        return self.python_version

    @staticmethod
    def _resolve_python_version() -> sys.version_info:
        return _VersionInfo(*sys.version_info[:3])
