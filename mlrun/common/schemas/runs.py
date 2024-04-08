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

import pydantic

from .constants import OrderType, RunPartitionByField


class RunIdentifier(pydantic.BaseModel):
    kind: typing.Literal["run"] = "run"
    uid: typing.Optional[str]
    iter: typing.Optional[int]


class ListRunsRequest(pydantic.BaseModel):
    class Config:
        allow_population_by_field_name = True

    project: typing.Optional[str]
    name: typing.Optional[str]
    uid: typing.Optional[str]
    project: typing.Optional[str]
    labels: typing.Optional[str]
    state: typing.Optional[str]
    last: typing.Optional[str]
    sort: typing.Optional[str]
    iter: typing.Optional[int]
    start_time_from: typing.Optional[str]
    start_time_to: typing.Optional[str]
    last_update_time_from: typing.Optional[str]
    last_update_time_to: typing.Optional[str]
    partition_by: typing.Optional[RunPartitionByField] = pydantic.Field(
        None, alias="partition-by"
    )
    rows_per_partition: typing.Optional[int] = pydantic.Field(
        1, alias="rows-per-partition", gt=0
    )
    partition_sort_by: typing.Optional[str] = pydantic.Field(
        None, alias="partition-sort-by"
    )
    partition_order: typing.Optional[OrderType] = pydantic.Field(
        OrderType.desc, alias="partition-order"
    )
    max_partitions: typing.Optional[int] = pydantic.Field(
        0, alias="max-partitions", ge=0
    )
    with_notifications: typing.Optional[bool] = pydantic.Field(
        None, alias="with-notifications"
    )
