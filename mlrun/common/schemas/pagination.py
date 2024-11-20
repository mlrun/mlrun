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


class PaginationInfo(pydantic.v1.BaseModel):
    class Config:
        allow_population_by_field_name = True

    page: typing.Optional[int]
    page_size: typing.Optional[int] = pydantic.v1.Field(alias="page-size")
    page_token: typing.Optional[str] = pydantic.v1.Field(alias="page-token")
