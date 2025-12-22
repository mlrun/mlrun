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

from pathlib import Path
from typing import Optional

import yaml

from mlrun.common.schemas.hub import HubSourceType
from mlrun.run import get_object

from ..errors import MLRunInvalidArgumentError
from ..utils import extend_hub_uri_if_needed
from .base import HubAsset


class HubStep(HubAsset):
    ASSET_TYPE = HubSourceType.steps

    def __init__(
        self,
        name: str,
        version: str,
        class_name: str,
        default_handler: str,
        description: Optional[str] = None,
        categories: Optional[list] = None,
        requirements: Optional[list] = None,
        local_path: Optional[Path] = None,
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
        self.class_name = class_name
        self.default_handler = default_handler

    def download_files(
        self,
        local_path: Optional[str] = None,
        download_example: bool = False,
    ):
        """
        Download this step's code file.
        :param local_path: Target directory to download the step files to. Defaults to the current working directory.
                           This path will be used to locate the code file when importing it as a python module.
        :param download_example: Whether to download the example notebook if available. Defaults to False.
        """
        super().download_files(
            local_path=local_path,
            download_example=download_example,
        )


def get_hub_step(
    url: str,
    local_path: Optional[str] = None,
    download_files: bool = True,
    include_example: bool = False,
) -> HubStep:
    """
    Get a hub-step object containing metadata of the requested step.
    :param url: Hub step url in the format "hub://[<source>/]<item-name>[:<tag>]"
    :param local_path: Path to target directory for the step files. Ignored when download_files is set to False.
                       Defaults to the current working directory.
    :param download_files: When set to True, the step code files are downloaded
    :param include_example: When set to True, the example notebook will also be downloaded (ignored if download_files is
                           False)

    :return: HubStep object
    """
    item_yaml_url, is_hub_uri = extend_hub_uri_if_needed(
        uri=url, asset_type=HubSourceType.steps, file="item.yaml"
    )
    if not is_hub_uri:
        raise MLRunInvalidArgumentError("Not a valid hub URL")
    yaml_obj = get_object(url=item_yaml_url)
    item_yaml = yaml.safe_load(yaml_obj)
    spec = item_yaml.pop("spec", {})
    class_name = item_yaml.pop("className", "")
    default_handler = item_yaml.pop("defaultHandler", "")
    hub_step = HubStep(
        **item_yaml,
        **spec,
        class_name=class_name,
        default_handler=default_handler,
        url=url,
    )
    if download_files:
        hub_step.download_files(local_path=local_path, download_example=include_example)
    return hub_step
