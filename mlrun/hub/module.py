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

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import yaml

import mlrun.common.types
import mlrun.utils
from mlrun.common.schemas.hub import HubModuleType, HubSourceType
from mlrun.run import function_to_module, get_object
from mlrun.utils import logger

from ..errors import MLRunBadRequestError
from ..model import ModelObj
from ..utils import extend_hub_uri_if_needed


class HubModule(ModelObj):
    def __init__(
        self,
        name: str,
        kind: Union[HubModuleType, str],
        version: Optional[str] = None,
        description: Optional[str] = None,
        categories: Optional[list] = None,
        requirements: Optional[list] = None,
        local_path: Optional[str] = None,
        filename: Optional[str] = None,
        example: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs,  # catch all for unused args
    ):
        self.name: str = name
        self.version: str = version
        self.kind: HubModuleType = kind
        self.description: str = description or ""
        self.categories: list = categories or []
        self.requirements: list = requirements or []
        self.local_path: str = local_path or ""
        self.filename: str = filename or name + ".py"
        self.example: str = example or ""
        self.url: str = url or ""

    def module(self):
        """Import the module after downloading its fils to local_path"""
        try:
            return function_to_module(code=self.filename, workdir=self.local_path)
        except FileNotFoundError:
            searched_path = self.local_path or "./"
            raise FileNotFoundError(
                f"Module file {self.filename} not found in {searched_path}, try calling download_module_files() first"
            )

    def install_requirements(self) -> None:
        """
        Install pip-style requirements (e.g., ["pandas>=2.0.0", "requests==2.31.0"]).
        """
        for req in self.requirements:
            logger.info(f"Installing {req} ...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", req], check=True, text=True
                )
                logger.info(f"Installed {req}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {req} (exit code {e.returncode})")

    def download_module_files(self, local_path=None, secrets=None):
        """
        Download this hub module’s files (code file and, if available, an example notebook) to the target directory
        specified by `local_path` (defaults to the current working directory).
        This path will be used later to locate the code file when importing the module.
        """
        self.local_path = self.verify_directory(path=local_path)
        source_url, _ = extend_hub_uri_if_needed(
            uri=self.url, asset_type=HubSourceType.modules, file=self.filename
        )
        self._download_object(
            obj_url=source_url, target_name=self.filename, secrets=secrets
        )
        if self.example:
            example_url, _ = extend_hub_uri_if_needed(
                uri=self.url, asset_type=HubSourceType.modules, file=self.example
            )
            self._download_object(
                obj_url=example_url, target_name=self.example, secrets=secrets
            )

    def _download_object(self, obj_url, target_name, secrets=None):
        data = get_object(url=obj_url, secrets=secrets)
        target_dir = self.local_path if self.local_path is not None else os.getcwd()
        target_filepath = os.path.join(target_dir, target_name)
        with open(target_filepath, "wb") as f:
            f.write(data)

    @staticmethod
    def verify_directory(path: Optional[str] = None) -> Path:
        """
        Validate that the given path is an existing directory.
        If no path has been provided, returns current working directory.
        """
        if path:
            path = Path(path)
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")
            return path
        return Path(os.getcwd())

    def get_module_file_path(self):
        if not self.local_path:
            raise MLRunBadRequestError(
                "module files haven't been downloaded yet, try calling download_module_files() first"
            )
        return str(Path(self.local_path) / self.filename)


def get_hub_module(
    url: str = "",
    download_files: Optional[bool] = True,
    secrets: Optional[dict] = None,
    local_path: Optional[str] = None,
) -> HubModule:
    """
    Get a hub-module object containing metadata of the requested module.
    :param url: Hub module url in the format "hub://[<source>/]<item-name>[:<tag>]"
    :param download_files: When set to True, the module files (code file and example notebook) are downloaded
    :param secrets: Optional, credentials dict for DB or URL (s3, v3io, ...)
    :param local_path: Path to target directory for the module files. Ignored when download_files is set to False.
                       Defaults to the current working directory.

    :return: HubModule object
    """
    item_yaml_url, is_hub_uri = extend_hub_uri_if_needed(
        uri=url, asset_type=HubSourceType.modules, file="item.yaml"
    )
    if not is_hub_uri:
        raise mlrun.errors.MLRunInvalidArgumentError("Not a valid hub URL")
    yaml_obj = get_object(url=item_yaml_url, secrets=secrets)
    item_yaml = yaml.safe_load(yaml_obj)
    spec = item_yaml.pop("spec", {})
    hub_module = HubModule(**item_yaml, **spec, url=url)
    if download_files:
        hub_module.download_module_files(local_path=local_path, secrets=secrets)
    return hub_module


def import_module(url="", install_requirements=False, secrets=None, local_path=None):
    """
    Import a module from the hub to use directly.
    :param url: hub module url in the format "hub://[<source>/]<item-name>[:<tag>]"
    :param install_requirements: when set to True, the module's requirements are installed.
    :param secrets: optional, credentials dict for DB or URL (s3, v3io, ...)
    :param local_path: Path to target directory for the module files (code and example notebook).
                       Defaults to the current working directory.

    :return: the module
    """
    hub_module: HubModule = get_hub_module(
        url=url, download_files=True, secrets=secrets, local_path=local_path
    )
    if install_requirements:
        hub_module.install_requirements()
    return hub_module.module()
