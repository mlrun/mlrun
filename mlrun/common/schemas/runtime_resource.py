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

import mlrun.common.types


class ListRuntimeResourcesGroupByField(mlrun.common.types.StrEnum):
    job = "job"
    project = "project"


class RuntimeResource(pydantic.v1.BaseModel):
    name: str
    labels: dict[str, str] = {}
    status: typing.Optional[dict]


class RuntimeResources(pydantic.v1.BaseModel):
    crd_resources: list[RuntimeResource] = []
    pod_resources: list[RuntimeResource] = []
    # only for dask runtime
    service_resources: typing.Optional[list[RuntimeResource]] = None

    class Config:
        extra = pydantic.v1.Extra.allow


class KindRuntimeResources(pydantic.v1.BaseModel):
    kind: str
    resources: RuntimeResources


RuntimeResourcesOutput = list[KindRuntimeResources]


# project name -> job uid -> runtime resources
GroupedByJobRuntimeResourcesOutput = dict[str, dict[str, RuntimeResources]]
# project name -> kind -> runtime resources
GroupedByProjectRuntimeResourcesOutput = dict[str, dict[str, RuntimeResources]]
