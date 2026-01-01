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
from typing import ClassVar, Optional

from mlrun.common.schemas.hub import HubSourceType
from mlrun.run import function_to_module, get_object
from mlrun.utils import logger

from ..errors import MLRunBadRequestError
from ..model import ModelObj
from ..utils import extend_hub_uri_if_needed


class HubAsset(ModelObj):
    ASSET_TYPE: ClassVar[HubSourceType]

    def __init__(
        self,
        name: str,
        version: str,
        description: Optional[str] = None,
        categories: Optional[list] = None,
        requirements: Optional[list] = None,
        local_path: Optional[Path] = None,
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
        self.local_path: Optional[Path] = local_path
        self.filename: str = filename or name
        self.example: str = example or ""
        self.url: str = url or ""

    def module(self):
        """Import the code of the asset as a module."""
        try:
            return function_to_module(code=self.filename, workdir=self.local_path)
        except Exception as e:
            raise MLRunBadRequestError(
                f"Failed to import module from {self.get_src_file_path()}: {e}"
            )

    def install_requirements(self) -> None:
        """
        Install pip-style requirements of the asset (e.g., ["pandas>=2.0.0", "requests==2.31.0"]).
        """
        if not self.requirements or len(self.requirements) == 0:
            logger.info("No requirements to install.")
            return
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
        local_path: Optional[str] = None,
        download_example: bool = True,
    ):
        """
        Download this hub asset’s files (code file and, if available and requested, an example notebook) to the target
        directory specified by `local_path` (defaults to the current working directory).
        This path will be used later to locate the code file when calling module().
        """
        self.local_path = self.verify_directory(path=local_path)
        source_url, _ = extend_hub_uri_if_needed(
            uri=self.url, asset_type=self.ASSET_TYPE, file=self.filename
        )
        self._download_object(obj_url=source_url, target_name=self.filename)
        if download_example and self.example:
            example_url, _ = extend_hub_uri_if_needed(
                uri=self.url, asset_type=self.ASSET_TYPE, file=self.example
            )
            self._download_object(obj_url=example_url, target_name=self.example)

    def _download_object(self, obj_url, target_name, secrets=None):
        data = get_object(url=obj_url, secrets=secrets)
        target_filepath = os.path.join(self.local_path, target_name)
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

    def get_src_file_path(self) -> str:
        """Get the full path to the asset's code file."""
        if not self.local_path:
            raise MLRunBadRequestError(
                f"Local path not set. Call download_files() first to download the asset files, or "
                f"set_local_path() with the directory containing {self.filename}"
            )
        src_path = Path(self.local_path) / self.filename
        if not src_path.exists():
            raise FileNotFoundError(
                f"File {self.filename} not found in {self.local_path}. Call download_files() first to download the "
                f"asset files, or set_local_path() with the directory containing {self.filename}"
            )

        return str(src_path)

    def set_local_path(self, path: str):
        """Set the local path where the asset's files are stored."""
        self.local_path = self.verify_directory(path=path)
