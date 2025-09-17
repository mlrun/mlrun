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
from ..model import ModelObj
from ..utils import extend_hub_uri_if_needed
import mlrun.common.types
from mlrun.run import get_object
from mlrun.common.schemas.hub import HubSourceType

class ModuleType(mlrun.common.types.StrEnum):
    generic = "generic"
    monitoring_app = "monitoring-app"

class HubModule(ModelObj):
    def __init__(
            self,
            name: Optional[str] = "",
            version: Optional[str] = "",
            kind: Optional[Union[ModuleType, str]] = None,
            description: Optional[str] = "",
            requirements: Optional[list] = None,
            **kwargs
    ):
        self.name: str = name
        self.version: str = version
        self.kind: ModuleType = kind
        self.description: str = description
        self.requirements: list = requirements or []

    def module(self):
        # TODO: implement
        pass

    def install_requirements(self):
        # TODO: implement
        pass

def _download_object(url: str, filename: str, local_path=None, secrets=None):
    data = get_object(url, secrets=secrets)
    target_dir = local_path if local_path is not None else os.getcwd()
    os.makedirs(target_dir, exist_ok=True) # create directory if missing # TODO: want this?
    target_filepath = os.path.join(target_dir, filename)
    with open(target_filepath, "wb") as f:
        f.write(data)

def _dowlnload_module_files(url, item_yaml, secrets=None, local_path=None):
    name = item_yaml.get("name", "")
    filename = f"{name}.py" # assume single file with module name
    source_url, _ = extend_hub_uri_if_needed(url, HubSourceType.modules, filename)
    _download_object(source_url, filename, local_path, secrets)
    if item_yaml.get("example", ""):
        filename = item_yaml.get("example")
        example_url, _ = extend_hub_uri_if_needed(url, HubSourceType.modules, filename)
        _download_object(example_url, filename, local_path, secrets)

def get_hub_module(url="", secrets=None, local_path=None):
    item_yaml_url, is_hub_uri = extend_hub_uri_if_needed(url, HubSourceType.modules, "item.yaml")
    if not is_hub_uri:
        raise mlrun.errors.MLRunInvalidArgumentError("Not a valid hub uri")
    yaml_obj = get_object(item_yaml_url, secrets)
    item_yaml = yaml.safe_load(yaml_obj)
    spec = item_yaml.pop("spec", {})
    hub_module = HubModule(**item_yaml, **spec)
    _dowlnload_module_files(url, item_yaml, secrets, local_path)
    return hub_module

def import_module():
    hub_module: HubModule = get_hub_module() # also downloads the files
    return hub_module.module() # import the mo ule and return it