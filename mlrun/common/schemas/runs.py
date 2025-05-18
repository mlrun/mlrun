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

import pydantic.v1
from deprecated import deprecated

import mlrun.common.types


class RunIdentifier(pydantic.v1.BaseModel):
    kind: typing.Literal["run"] = "run"
    uid: typing.Optional[str]
    iter: typing.Optional[int]


@deprecated(
    version="1.7.0",
    reason="mlrun.common.schemas.RunsFormat is deprecated and will be removed in 1.10.0. "
    "Use mlrun.common.formatters.RunFormat instead.",
    category=FutureWarning,
)
class RunsFormat(mlrun.common.types.StrEnum):
    # No enrichment, data is pulled as-is from the database.
    standard = "standard"

    # Performs run enrichment, including the run's artifacts. Only available for the `get` run API.
    full = "full"
