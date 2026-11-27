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

import typing

import pydantic.v1

from .object import ObjectStatus


class ArtifactIdentifier(pydantic.v1.BaseModel):
    # artifact kind
    kind: str | None
    key: str | None
    iter: int | None
    uid: str | None
    producer_id: str | None
    # TODO support hash once saved as a column in the artifacts table
    # hash: typing.Optional[str]


class ArtifactMetadata(pydantic.v1.BaseModel):
    key: str
    project: str
    iter: int | None
    tree: str | None
    tag: str | None

    class Config:
        extra = pydantic.v1.Extra.allow


class ArtifactSpec(pydantic.v1.BaseModel):
    src_path: str | None
    target_path: str | None
    viewer: str | None
    inline: str | None
    size: int | None
    db_key: str | None
    extra_data: dict[str, typing.Any] | None
    unpackaging_instructions: dict[str, typing.Any] | None
    parent_uri: str | None

    class Config:
        extra = pydantic.v1.Extra.allow


class Artifact(pydantic.v1.BaseModel):
    kind: str
    metadata: ArtifactMetadata
    spec: ArtifactSpec
    status: ObjectStatus
