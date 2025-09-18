# Copyright 2025 Iguazio
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

import yaml
import os
from typing import Optional, Union

import mlrun.utils
from ..model import ModelObj
from ..utils import extend_hub_uri_if_needed
import mlrun.common.types
from mlrun.run import get_object, function_to_module
from mlrun.common.schemas.hub import HubSourceType
from pydantic import DirectoryPath, TypeAdapter

_DIR = TypeAdapter(DirectoryPath)

class ModuleType(mlrun.common.types.StrEnum):
    generic = "generic"
    monitoring_app = "monitoring-app"

class HubModule(ModelObj):
    def __init__(
            self,
            name: str,
            kind: Union[ModuleType, str],
            version: Optional[str] = None,
            description: Optional[str] = None,
            categories: Optional[list] = None,
            requirements: Optional[list] = None,
            local_path: Optional[str] = None,
            filename: Optional[str] = None,
            example: Optional[str] = None,
            url: Optional[str] = None,
            **kwargs # catch all for unused args
    ):
        self.name: str = name
        self.version: str = version
        self.kind: ModuleType = kind
        self.description: str = description or ""
        self.categories: list = categories or []
        self.requirements: list = requirements or []
        self.local_path: str = local_path or ""
        self.filename: str = filename or name+".py"
        self.example: str = example or ""
        self.url: str = url or ""

    # @staticmethod
    # def _validate_path(self, path_value) -> str:
    #     if path_value is None:
    #         return None
    #     try:
    #         path = _DIR.validate_python(path_value) # raise error if doesn't exist
    #         return str(path)
    #     except Exception as exc:
    #         raise mlrun.errors.MLRunInvalidArgumentError(f"Invalid local_path value {path_value}, error: {exc}") from exc
    # @property
    # def local_path(self) -> str:
    #     return self.local_path
    #
    # @local_path.setter
    # def local_path(self, value) -> None:
    #     self.local_path = self._validate_path(value)

    def module(self):
        try:
            return function_to_module(self.filename, self.local_path)
        except FileNotFoundError:
            mlrun.utils.logger.warning(f"Module file {self.filename} not found in {self.local_path}, try calling download_module_files()")
            return None

    def install_requirements(self):
        # TODO: implement
        pass

    def download_module_files(self, local_path=None, secrets=None):
        self.local_path = local_path
        source_url, _ = extend_hub_uri_if_needed(self.url, HubSourceType.modules, self.filename)
        self._download_object(source_url, self.filename, secrets)
        if self.example:
            example_url, _ = extend_hub_uri_if_needed(self.url, HubSourceType.modules, self.example)
            self._download_object(example_url, self.example, secrets)

    def _download_object(self, obj_url, target_name, secrets=None):
        data = get_object(obj_url, secrets=secrets)
        target_dir = self.local_path if self.local_path is not None else os.getcwd()
        target_filepath = os.path.join(target_dir, target_name)
        with open(target_filepath, "wb") as f:
            f.write(data)


def get_hub_module(url="", download_files=True, secrets=None, local_path=None):
    item_yaml_url, is_hub_uri = extend_hub_uri_if_needed(url, HubSourceType.modules, "item.yaml")
    if not is_hub_uri:
        raise mlrun.errors.MLRunInvalidArgumentError(f"Not a valid hub URL")
    yaml_obj = get_object(item_yaml_url, secrets)
    item_yaml = yaml.safe_load(yaml_obj)
    spec = item_yaml.pop("spec", {})
    hub_module = HubModule(**item_yaml, **spec, url=url)
    if download_files:
        hub_module.download_module_files(local_path, secrets)
    return hub_module

def import_module(url="", secrets=None, local_path=None):
    hub_module: HubModule = get_hub_module(url, True, secrets, local_path)
    return hub_module.module()