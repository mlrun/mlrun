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

import pydantic

from .object import ObjectStatus


class ArtifactIdentifier(pydantic.BaseModel):
    # artifact kind
    kind: str | None = None
    key: str | None = None
    iter: int | None = None
    uid: str | None = None
    producer_id: str | None = None
    # TODO support hash once saved as a column in the artifacts table
    # hash: typing.Optional[str]


class ArtifactMetadata(pydantic.BaseModel):
    key: str
    project: str
    iter: int | None = None
    tree: str | None = None
    tag: str | None = None

    model_config = pydantic.ConfigDict(extra="allow")


class ArtifactSpec(pydantic.BaseModel):
    src_path: str | None = None
    target_path: str | None = None
    viewer: str | None = None
    inline: str | None = None
    size: int | None = None
    db_key: str | None = None
    extra_data: dict[str, typing.Any] | None = None
    unpackaging_instructions: dict[str, typing.Any] | None = None
    parent_uri: str | None = None

    model_config = pydantic.ConfigDict(extra="allow")


class Artifact(pydantic.BaseModel):
    kind: str
    metadata: ArtifactMetadata
    spec: ArtifactSpec
    status: ObjectStatus
