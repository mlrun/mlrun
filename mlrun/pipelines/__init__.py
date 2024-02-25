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

import sys
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location

import kfp_server_api


def resolve_pipeline_engine():
    if kfp_server_api.__version__.startswith("2.0"):
        return "kfp-v2.0"
    return "kfp-v1.8"


PIPELINE_COMPATIBILITY_MODE = resolve_pipeline_engine()
_pipeline_module_locations = {
    "kfp-v1.8": "mlrun/pipelines/kfp/v1_8",
    "kfp-v2.0": "mlrun/pipelines/kfp/v2_0",
}


class PipelineEngineModuleFinder(MetaPathFinder):
    @staticmethod
    def _resolve_module_path(fullname, path):
        path_prefix = path[0].replace(
            "mlrun/pipelines",
            _pipeline_module_locations.get(PIPELINE_COMPATIBILITY_MODE),
        )
        path_suffix = fullname.replace("mlrun.pipelines", "")
        if path_suffix:
            return f"{path_prefix}{path_suffix.replace('.', '/')}.py"
        return path_prefix + "/__init__.py"

    def find_spec(self, fullname, path, target=None):
        # TODO: improve this condition
        if "mlrun.pipelines" in fullname and "mlrun.pipelines.common" not in fullname:
            return spec_from_file_location(
                fullname, self._resolve_module_path(fullname, path)
            )
        return None  # don't interfere with the current import request


sys.meta_path.insert(0, PipelineEngineModuleFinder())
