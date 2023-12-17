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

import pydantic

import mlrun.common.types

from .object import ObjectStatus


class ArtifactCategories(mlrun.common.types.StrEnum):
    model = "model"
    dataset = "dataset"
    other = "other"

    # we define the link as a category to prevent import cycles, but it's not a real category
    # and should not be used as such
    link = "link"

    def to_kinds_filter(self) -> typing.Tuple[typing.List[str], bool]:
        link_kind = ArtifactCategories.link.value

        if self.value == ArtifactCategories.model.value:
            return [ArtifactCategories.model.value, link_kind], False
        if self.value == ArtifactCategories.dataset.value:
            return [ArtifactCategories.dataset.value, link_kind], False
        if self.value == ArtifactCategories.other.value:
            return (
                [
                    ArtifactCategories.model.value,
                    ArtifactCategories.dataset.value,
                ],
                True,
            )


class ArtifactIdentifier(pydantic.BaseModel):
    # artifact kind
    kind: typing.Optional[str]
    key: typing.Optional[str]
    iter: typing.Optional[int]
    uid: typing.Optional[str]
    producer_id: typing.Optional[str]
    # TODO support hash once saved as a column in the artifacts table
    # hash: typing.Optional[str]


class ArtifactsFormat(mlrun.common.types.StrEnum):
    # TODO: add a format that returns a minimal response
    full = "full"


class ArtifactMetadata(pydantic.BaseModel):
    key: str
    project: str
    iter: typing.Optional[int]
    tree: typing.Optional[str]
    tag: typing.Optional[str]

    class Config:
        extra = pydantic.Extra.allow


class ArtifactSpec(pydantic.BaseModel):
    src_path: typing.Optional[str]
    target_path: typing.Optional[str]
    viewer: typing.Optional[str]
    inline: typing.Optional[str]
    size: typing.Optional[int]
    db_key: typing.Optional[str]
    extra_data: typing.Optional[typing.Dict[str, typing.Any]]
    unpackaging_instructions: typing.Optional[typing.Dict[str, typing.Any]]

    class Config:
        extra = pydantic.Extra.allow


class Artifact(pydantic.BaseModel):
    kind: str
    metadata: ArtifactMetadata
    spec: ArtifactSpec
    status: ObjectStatus
