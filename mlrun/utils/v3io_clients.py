# Copyright 2023 Iguazio
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
import threading

from v3io.dataplane import Client as V3IOClient
from v3io_frames import Client as get_client
from v3io_frames.client import ClientBase

_frames_thread_local_clients = threading.local()
_v3io_thread_local_clients = threading.local()


def get_frames_client(**kwargs) -> ClientBase:
    global _frames_thread_local_clients

    if not hasattr(_frames_thread_local_clients, "clients"):
        _frames_thread_local_clients.clients = {}

    clients = _frames_thread_local_clients.clients

    kw_set = frozenset(kwargs.items())
    if kw_set not in clients:
        clients[kw_set] = get_client(**kwargs)

    return clients[kw_set]


def get_v3io_client(**kwargs) -> V3IOClient:
    global _v3io_thread_local_clients

    if not hasattr(_v3io_thread_local_clients, "clients"):
        _v3io_thread_local_clients.clients = {}

    clients = _v3io_thread_local_clients.clients

    kw_set = frozenset(kwargs.items())
    if kw_set not in clients:
        clients[kw_set] = V3IOClient(**kwargs)

    return clients[kw_set]
