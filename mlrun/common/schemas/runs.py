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

import typing

# TODO: When we remove support for python 3.7, we can use Literal from the typing package.
#       Remove the following try/except block with import from typing_extensions.
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pydantic


class RunIdentifier(pydantic.BaseModel):
    kind: Literal["run"] = "run"
    uid: typing.Optional[str]
    iter: typing.Optional[int]
