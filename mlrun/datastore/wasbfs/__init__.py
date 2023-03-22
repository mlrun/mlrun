# Copyright 2020 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""wasbfs - An fsspec driver for wasb/wasbs"""

__all__ = [
    "WasbFS",
]


import fsspec

from .fs import WasbFS

if hasattr(fsspec, "register_implementation"):
    # TODO: Not sure about clobber=True
    fsspec.register_implementation("wasbs", WasbFS, clobber=True)
    fsspec.register_implementation("wasb", WasbFS, clobber=True)
else:
    from fsspec.registry import known_implementations

    known_implementations["wasbs"] = {
        "class": "wasbfs.WasbFS",
        "err": "Please install WasbFS to use the wasb fileysstem class",
    }
    known_implementations["wasb"] = {
        "class": "wasbfs.WasbFS",
        "err": "Please install WasbFS to use the wasb fileysstem class",
    }

    del known_implementations

del fsspec  # clear the module namespace
