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
#

import inspect
from collections.abc import Iterator, MutableMapping
from typing import Any, NoReturn

project_annotation = "mlrun/project"
run_annotation = "mlrun/pipeline-step-type"
function_annotation = "mlrun/function-uri"


class FlexibleMapper(MutableMapping):
    _external_data: dict

    def __init__(self, external_data: Any):
        if isinstance(external_data, dict):
            self._external_data = external_data
        elif hasattr(external_data, "to_dict"):
            self._external_data = external_data.to_dict()

    # TODO: decide if we should kill the dict compatibility layer on get, set and del
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return self._external_data[key]

    def __setitem__(self, key, value) -> NoReturn:
        try:
            setattr(self, key, value)
        except AttributeError:
            self._external_data[key] = value

    def __delitem__(self, key) -> NoReturn:
        try:
            delattr(self, key)
        except AttributeError:
            del self._external_data[key]

    def __len__(self) -> int:
        # TODO: review the intrinsic responsibilities of __len__ on MutableMapping to ensure full compatibility
        return len(self._external_data) + len(vars(self)) - 1

    def __iter__(self) -> Iterator[str]:
        yield from [
            m[0]
            for m in inspect.getmembers(self)
            if not callable(m[1]) and not m[0].startswith("_")
        ]

    def __bool__(self) -> bool:
        return bool(self._external_data)

    def to_dict(self) -> dict:
        return {k: v for k, v in self}
