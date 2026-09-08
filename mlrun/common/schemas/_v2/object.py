# Copyright 2026 Iguazio
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

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ObjectMetadata(BaseModel):
    name: str
    project: str | None = None
    tag: str | None = None
    labels: dict | None = {}
    updated: datetime | None = None
    created: datetime | None = None
    uid: str | None = None

    model_config = ConfigDict(extra="allow")


class ObjectStatus(BaseModel):
    state: str | None = None

    model_config = ConfigDict(extra="allow")


class ObjectSpec(BaseModel):
    model_config = ConfigDict(extra="allow")


class LabelRecord(BaseModel):
    id: int
    name: str
    value: str

    model_config = ConfigDict(from_attributes=True)


class ObjectRecord(BaseModel):
    id: int
    name: str
    project: str
    uid: str
    updated: datetime | None = None
    labels: list[LabelRecord]
    # state is extracted from the full status dict to enable queries
    state: str | None = None
    full_object: dict | None = None

    model_config = ConfigDict(from_attributes=True)
