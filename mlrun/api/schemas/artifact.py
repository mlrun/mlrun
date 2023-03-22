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


class ArtifactCategories(mlrun.api.utils.helpers.StrEnum):
    model = "model"
    dataset = "dataset"
    other = "other"

    def to_kinds_filter(self) -> typing.Tuple[typing.List[str], bool]:
        # FIXME: these artifact definitions (or at least the kinds enum) should sit in a dedicated module
        # import here to prevent import cycle
        import mlrun.artifacts.dataset
        import mlrun.artifacts.model

        link_kind = mlrun.artifacts.base.LinkArtifact.kind

        if self.value == ArtifactCategories.model.value:
            return [mlrun.artifacts.model.ModelArtifact.kind, link_kind], False
        if self.value == ArtifactCategories.dataset.value:
            return [mlrun.artifacts.dataset.DatasetArtifact.kind, link_kind], False
        if self.value == ArtifactCategories.other.value:
            return (
                [
                    mlrun.artifacts.model.ModelArtifact.kind,
                    mlrun.artifacts.dataset.DatasetArtifact.kind,
                ],
                True,
            )


class ArtifactIdentifier(pydantic.BaseModel):
    # artifact kind
    kind: typing.Optional[str]
    key: typing.Optional[str]
    iter: typing.Optional[int]
    uid: typing.Optional[str]
    # TODO support hash once saved as a column in the artifacts table
    # hash: typing.Optional[str]


class ArtifactsFormat(mlrun.api.utils.helpers.StrEnum):
    full = "full"
    legacy = "legacy"
