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
from pathlib import Path
import subprocess
import sys


import mlrun.utils
from ..model import ModelObj
from ..utils import extend_hub_uri_if_needed
import mlrun.common.types
from mlrun.run import get_object, function_to_module
from mlrun.common.schemas.hub import HubSourceType

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


    def module(self):
        try:
            return function_to_module(self.filename, self.local_path)
        except FileNotFoundError:
            searched_path = self.local_path or "./"
            mlrun.utils.logger.warning(f"Module file {self.filename} not found in {searched_path}, try calling download_module_files() first")
            return None

    def install_requirements(self) -> None:
        """
        Install pip-style requirements (e.g., ["pandas>=2.0.0", "requests==2.31.0"]).
        """
        for req in self.requirements:
            print(f"[INFO] Installing {req} ...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", req],
                    check=True, text=True
                )
                print(f"[SUCCESS] Installed {req}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to install {req} (exit code {e.returncode})")

    def download_module_files(self, local_path=None, secrets=None):
        self.local_path = self.verify_directory(local_path)
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

    @staticmethod
    def verify_directory(path) -> Path:
        """Validate that the given path is an existing directory."""
        if path:
            path = Path(path)
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")
        return path

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

def import_module(url="", install_requirements=False, secrets=None, local_path=None):
    hub_module: HubModule = get_hub_module(url, True, secrets, local_path)
    if install_requirements:
        hub_module.install_requirements()
    return hub_module.module()