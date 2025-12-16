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
from typing import Optional

from mlrun.common.schemas.hub import HubSourceType
from mlrun.run import function_to_module, get_object
from mlrun.utils import logger

from ..errors import MLRunBadRequestError
from ..model import ModelObj
from ..utils import extend_hub_uri_if_needed


class HubAsset(ModelObj):
    def __init__(
        self,
        name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        categories: Optional[list] = None,
        requirements: Optional[list] = None,
        local_path: Optional[str] = None,
        filename: Optional[str] = None,
        example: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs,
    ):
        self.name: str = name
        self.version: str = version
        self.description: str = description or ""
        self.categories: list = categories or []
        self.requirements: list = requirements or []
        self.local_path: str = local_path or ""
        self.filename: str = filename or name
        self.example: str = example or ""
        self.url: str = url or ""

    def module(self):
        """Import the code as a module"""
        try:
            return function_to_module(code=self.filename, workdir=self.local_path)
        except FileNotFoundError:
            searched_path = self.local_path or "./"
            raise FileNotFoundError(
                f"Item file {self.filename} not found in {searched_path}, try calling download_files() first"
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

    def download_files(
        self,
        asset_type: HubSourceType,
        local_path: str = None,
        download_example: bool = True,
    ):
        """
        Download this hub item’s files (code file and, if available and requested, an example notebook) to the target directory
        specified by `local_path` (defaults to the current working directory).
        This path will be used later to locate the code file when calling module().
        """
        self.local_path = self.verify_directory(path=local_path)
        source_url, _ = extend_hub_uri_if_needed(
            uri=self.url, asset_type=asset_type, file=self.filename
        )
        self._download_object(obj_url=source_url, target_name=self.filename)
        if download_example and self.example:
            example_url, _ = extend_hub_uri_if_needed(
                uri=self.url, asset_type=asset_type, file=self.example
            )
            self._download_object(obj_url=example_url, target_name=self.example)

    def _download_object(self, obj_url, target_name):
        data = get_object(url=obj_url)
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

    def get_src_file_path(self):
        if not self.local_path:
            raise MLRunBadRequestError(
                "Item files haven't been downloaded yet, try calling download_files() first"
            )
        return str(Path(self.local_path) / self.filename)
