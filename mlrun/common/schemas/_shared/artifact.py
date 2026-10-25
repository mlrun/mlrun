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

import mlrun.common.types


class ArtifactCategories(mlrun.common.types.StrEnum):
    model = "model"
    dataset = "dataset"
    document = "document"
    llm_prompt = "llm-prompt"
    other = "other"

    # we define the link as a category to prevent import cycles, but it's not a real category
    # and should not be used as such
    link = "link"

    def to_kinds_filter(self) -> tuple[list[str], bool]:
        link_kind = ArtifactCategories.link.value

        if self.value == ArtifactCategories.model.value:
            return [ArtifactCategories.model.value, link_kind], False
        if self.value == ArtifactCategories.dataset.value:
            return [ArtifactCategories.dataset.value, link_kind], False
        if self.value == ArtifactCategories.document.value:
            return [ArtifactCategories.document.value, link_kind], False
        if self.value == ArtifactCategories.llm_prompt.value:
            return [ArtifactCategories.llm_prompt.value, link_kind], False
        if self.value == ArtifactCategories.other.value:
            return (
                [
                    ArtifactCategories.model.value,
                    ArtifactCategories.dataset.value,
                    ArtifactCategories.document.value,
                    ArtifactCategories.llm_prompt.value,
                ],
                True,
            )

    @classmethod
    def from_kind(cls, kind: str) -> "ArtifactCategories":
        if kind in [
            cls.model.value,
            cls.dataset.value,
            cls.document.value,
            cls.llm_prompt.value,
        ]:
            return cls(kind)
        return cls.other

    @staticmethod
    def all():
        """Return all applicable artifact categories"""
        return [
            ArtifactCategories.model,
            ArtifactCategories.dataset,
            ArtifactCategories.document,
            ArtifactCategories.llm_prompt,
        ]


class ArtifactsDeletionStrategies(mlrun.common.types.StrEnum):
    """Artifacts deletion strategies types."""

    metadata_only = "metadata-only"
    """Only removes the artifact db record, leaving all related artifact data in-place"""

    data_optional = "data-optional"
    """Delete the artifact data of the artifact as a best-effort.
    If artifact data deletion fails still try to delete the artifact db record"""

    data_force = "data-force"
    """Delete the artifact data, and if cannot delete it fail the deletion
    and don’t delete the artifact db record"""
