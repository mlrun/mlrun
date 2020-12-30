from typing import Dict, FrozenSet

from v3io.dataplane import Client as V3IOClient
from v3io_frames import Client as FramesClient

_v3io_clients: Dict[FrozenSet, V3IOClient] = {}
_frames_clients: Dict[FrozenSet, FramesClient] = {}


# TODO: Fix type hints for frames and v3io clients


def get_frames_client(**kwargs):
    global _frames_clients
    kw_set = frozenset(kwargs.items())
    if kw_set not in _frames_clients:
        _frames_clients[kw_set] = FramesClient(**kwargs)

    return _frames_clients[kw_set]


def get_v3io_client(**kwargs):
    global _v3io_clients
    kw_set = frozenset(kwargs.items())
    if kw_set not in _v3io_clients:
        _v3io_clients[kw_set] = V3IOClient(**kwargs)

    return _v3io_clients[kw_set]
