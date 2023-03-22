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
import typing

import pydantic

import mlrun.api.utils.helpers


class ListRuntimeResourcesGroupByField(mlrun.api.utils.helpers.StrEnum):
    job = "job"
    project = "project"


class RuntimeResource(pydantic.BaseModel):
    name: str
    labels: typing.Dict[str, str] = {}
    status: typing.Optional[typing.Dict]


class RuntimeResources(pydantic.BaseModel):
    crd_resources: typing.List[RuntimeResource] = []
    pod_resources: typing.List[RuntimeResource] = []
    # only for dask runtime
    service_resources: typing.Optional[typing.List[RuntimeResource]] = None

    class Config:
        extra = pydantic.Extra.allow


class KindRuntimeResources(pydantic.BaseModel):
    kind: str
    resources: RuntimeResources


RuntimeResourcesOutput = typing.List[KindRuntimeResources]


# project name -> job uid -> runtime resources
GroupedByJobRuntimeResourcesOutput = typing.Dict[
    str, typing.Dict[str, RuntimeResources]
]
# project name -> kind -> runtime resources
GroupedByProjectRuntimeResourcesOutput = typing.Dict[
    str, typing.Dict[str, RuntimeResources]
]
