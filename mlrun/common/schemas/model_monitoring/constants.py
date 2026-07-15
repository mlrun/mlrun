# Copyright 2024 Iguazio
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


# Environment-dispatched facade preserving the
# ``mlrun.common.schemas.model_monitoring.constants`` import path. Mirrors the full namespace of the underlying
# module(s) — public names plus the private helpers and re-exports callers rely
# on — so the submodule path stays byte-for-byte importable across the split.
from .._shared.model_monitoring import constants as _shared_mod

globals().update(
    {_n: _v for _n, _v in vars(_shared_mod).items() if not _n.startswith("__")}
)
__all__ = list(
    getattr(
        _shared_mod,
        "__all__",
        [_n for _n in vars(_shared_mod) if not _n.startswith("_")],
    )
)
del _shared_mod
