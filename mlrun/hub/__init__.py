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
from typing import Optional

import mlrun
from mlrun.common.schemas.hub import HubSourceType

from .module import get_hub_module, import_module
from .step import get_hub_step


def get_hub_item(
    source_name: str,
    item_name: str,
    version: Optional[str] = None,
    tag: Optional[str] = "latest",
    force_refresh: bool = False,
    item_type: HubSourceType = HubSourceType.functions,
) -> mlrun.common.schemas.hub.HubItem:
    """
    Retrieve a specific hub item.

    :param source_name: Name of source.
    :param item_name: Name of the item to retrieve, as it appears in the hub catalog.
    :param version: Get a specific version of the item. Default is ``None``.
    :param tag: Get a specific version of the item identified by tag. Default is ``latest``.
    :param force_refresh: Make the server fetch the information from the actual hub
        source, rather than
        rely on cached information. Default is ``False``.
    :param item_type: The type of item to retrieve from the hub source (e.g: functions, modules).
    :returns: :py:class:`~mlrun.common.schemas.hub.HubItem`.
    """
    db = mlrun.get_run_db()
    return db.get_hub_item(
        source_name=source_name,
        item_name=item_name,
        version=version,
        tag=tag,
        force_refresh=force_refresh,
        item_type=item_type,
    )
