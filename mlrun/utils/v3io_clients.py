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
#
from typing import Dict, FrozenSet

from v3io.dataplane import Client as V3IOClient
from v3io_frames import Client as get_client
from v3io_frames.client import ClientBase

_v3io_clients: Dict[FrozenSet, V3IOClient] = {}
_frames_clients: Dict[FrozenSet, ClientBase] = {}


def get_frames_client(**kwargs) -> ClientBase:
    global _frames_clients
    kw_set = frozenset(kwargs.items())
    if kw_set not in _frames_clients:
        _frames_clients[kw_set] = get_client(**kwargs)

    return _frames_clients[kw_set]


def get_v3io_client(**kwargs) -> V3IOClient:
    global _v3io_clients
    kw_set = frozenset(kwargs.items())
    if kw_set not in _v3io_clients:
        _v3io_clients[kw_set] = V3IOClient(**kwargs)

    return _v3io_clients[kw_set]
