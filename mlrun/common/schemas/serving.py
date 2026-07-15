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
# ``mlrun.common.schemas.serving`` import path. Mirrors the full namespace of the underlying
# module(s) — public names plus the private helpers and re-exports callers rely
# on — so the submodule path stays byte-for-byte importable across the split.
from ._dispatch import USE_V2_SCHEMAS as _USE_V2
from ._shared import serving as _shared_mod

if _USE_V2:
    from ._v2 import serving as _face_mod
else:
    from ._v1 import serving as _face_mod

globals().update(
    {_n: _v for _n, _v in vars(_shared_mod).items() if not _n.startswith("__")}
)
globals().update(
    {_n: _v for _n, _v in vars(_face_mod).items() if not _n.startswith("__")}
)
__all__ = [*_shared_mod.__all__, *_face_mod.__all__]
del _shared_mod, _face_mod
