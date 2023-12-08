# Copyright 2023 Iguazio
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

# TODO: Fetch currently installed KFP version to determine what package to enable
pipeline_compatibility_mode = "kfp-v1.8"


class PipelineEngineModuleFinder(MetaPathFinder):
    @staticmethod
    def __resolve_module_path(fullname, path):
        path_prefix = path[0].replace("mlrun/pipelines", "mlrun/pipelines/kfp/v1_8")
        path_suffix = fullname.replace("mlrun.pipelines", "")
        if path_suffix:
            return f"{path_prefix}{path_suffix.replace('.', '/')}.py"
        return path_prefix + "/__init__.py"

    def find_spec(self, fullname, path, target=None):
        if "mlrun.pipelines" in fullname:
            return spec_from_file_location(
                fullname, self.__resolve_module_path(fullname, path)
            )
        return None  # we don't need to participate on the current import


sys.meta_path.insert(0, PipelineEngineModuleFinder())
