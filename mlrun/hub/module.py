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

from typing import Optional, Union

import yaml
from deprecated import deprecated

import mlrun.common.types
import mlrun.utils
from mlrun.common.schemas.hub import HubModuleType, HubSourceType
from mlrun.run import get_object

from ..utils import extend_hub_uri_if_needed
from .base import HubAsset


class HubModule(HubAsset):
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
        super().__init__(
            name=name,
            version=version,
            description=description,
            categories=categories,
            requirements=requirements,
            local_path=local_path,
            filename=filename,
            example=example,
            url=url,
        )
        self.kind = kind

    # TODO: Remove this in 1.13.0
    @deprecated(
        version="1.11.0",
        reason="This function is deprecated and will be removed in 1.13. You can download module files by calling download_files() instead.",
        category=FutureWarning,
    )
    def download_module_files(self, local_path=None, secrets=None):
        """
        Download this hub module’s files (code file and, if available, an example notebook) to the target directory
        specified by `local_path` (defaults to the current working directory).
        This path will be used later to locate the code file when importing the module.
        """
        super().download_files(
            asset_type=HubSourceType.modules,
            local_path=local_path,
            download_example=True,
        )

    def download_files(self, local_path: str = None):
        super().download_files(
            asset_type=HubSourceType.modules,
            local_path=local_path,
            download_example=True,
        )

    def get_module_file_path(self):
        super().get_src_file_path()


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
        hub_module.download_files(local_path=local_path, secrets=secrets)
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
