# Copyright 2018 Iguazio
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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

__all__ = ["DataItem", "get_store_resource"]

from .datastore import StoreManager, uri_to_ipython, in_memory_store
from .base import DataItem
from .store_resources import is_store_uri, get_store_uri, get_store_resource
from .v3io import parse_v3io_path
from ..platforms.iguazio import OutputStream
from ..utils import logger

store_manager = StoreManager()


def set_in_memory_item(key, value):
    item = store_manager.object(f"memory://{key}")
    item.put(value)
    return item


def get_in_memory_items():
    return in_memory_store._items


def get_stream_pusher(stream_path: str, **kwargs):
    """get a stream pusher object from URL, currently only support v3io stream

    common kwargs::

        create:             create a new stream if doesnt exist
        shards:             number of shards
        retention_in_hours: stream retention in hours

    :param stream_path:        path/url of stream
    """

    if "://" not in stream_path:
        OutputStream(stream_path, **kwargs)
    elif stream_path.startswith("v3io"):
        endpoint, stream_path = parse_v3io_path(stream_path)
        OutputStream(stream_path, endpoint=endpoint, **kwargs)
    elif stream_path.startswith("dummy://"):
        return _DummyStream(**kwargs)
    else:
        raise ValueError(f"unsupported stream path {stream_path}")


class _DummyStream:
    """stream emulator for tests and debug"""

    def __init__(self, event_list=None, **kwargs):
        self.event_list = event_list or []

    def push(self, data):
        if not isinstance(data, list):
            data = [data]
        for item in data:
            logger.info(f"dummy stream got event: {item}")
            self.event_list.append(item)
