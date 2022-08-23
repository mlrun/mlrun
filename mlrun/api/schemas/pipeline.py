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
import enum
import typing

import pydantic


class PipelinesFormat(str, enum.Enum):
    full = "full"
    metadata_only = "metadata_only"
    summary = "summary"
    name_only = "name_only"


class PipelinesPagination(str):
    default_page_size = 20
    # https://github.com/kubeflow/pipelines/blob/master/backend/src/apiserver/list/list.go#L363
    max_page_size = 200


class PipelinesOutput(pydantic.BaseModel):
    # use the format query param to control what is returned
    runs: typing.List[typing.Union[dict, str]]
    total_size: int
    next_page_token: typing.Optional[str]
