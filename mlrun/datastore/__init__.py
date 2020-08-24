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

from .datastore import StoreManager, uri_to_ipython, get_object_stat, in_memory_store
from .base import DataItem


store_manager = StoreManager()


def set_in_memory_item(key, value):
    item = store_manager.object(f"memory://{key}")
    item.put(value)
    return item


def get_in_memory_items():
    return in_memory_store._items
