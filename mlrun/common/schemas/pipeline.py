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
#
import typing

import pydantic.v1
from deprecated import deprecated

import mlrun.common.types


@deprecated(
    version="1.7.0",
    reason="mlrun.common.schemas.PipelinesFormat is deprecated and will be removed in 1.10.0. "
    "Use mlrun.common.formatters.PipelineFormat instead.",
    category=FutureWarning,
)
class PipelinesFormat(mlrun.common.types.StrEnum):
    full = "full"
    metadata_only = "metadata_only"
    summary = "summary"
    name_only = "name_only"


class PipelinesPagination(str):
    default_page_size = 20
    # https://github.com/kubeflow/pipelines/blob/master/backend/src/apiserver/list/list.go#L363
    max_page_size = 200


class PipelinesOutput(pydantic.v1.BaseModel):
    # use the format query param to control what is returned
    runs: list[typing.Union[dict, str]]
    total_size: int
    next_page_token: typing.Optional[str]
