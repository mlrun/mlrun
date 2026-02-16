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

from mlrun import DataItem
from mlrun.package import DefaultPackager
from tests.package.assets import DummyDataItem


def test_pack_only_artifact_type():
    """
    Test that a packager with pack_summary but no unpack_summary correctly reports 'summary'
    as a packing-only artifact type.
    """

    class PackOnlyPackager(DefaultPackager):
        PACKABLE_OBJECT_TYPE = str
        DEFAULT_PACKING_ARTIFACT_TYPE = "summary"

        def pack_summary(self, obj: str, key: str) -> dict:
            return {key: f"summary: {obj}"}

    packager = PackOnlyPackager()

    # Packing artifact types should include "summary":
    assert "summary" in packager.get_supported_packing_artifact_types()

    # Unpacking artifact types should NOT include "summary":
    assert "summary" not in packager.get_supported_unpacking_artifact_types()

    # Union (get_supported_artifact_types) should include "summary":
    assert "summary" in packager.get_supported_artifact_types()

    # is_packable should return True for artifact_type="summary":
    assert packager.is_packable(obj="hello", artifact_type="summary") is True

    # is_unpackable should return False for artifact_type="summary":
    assert (
        packager.is_unpackable(
            data_item=DummyDataItem(key="test"),
            type_hint=str,
            artifact_type="summary",
        )
        is False
    )


def test_unpack_only_artifact_type():
    """
    Test that a packager with unpack_legacy but no pack_legacy correctly reports 'legacy'
    as an unpacking-only artifact type.
    """

    class UnpackOnlyPackager(DefaultPackager):
        PACKABLE_OBJECT_TYPE = str
        DEFAULT_PACKING_ARTIFACT_TYPE = "result"
        DEFAULT_UNPACKING_ARTIFACT_TYPE = "legacy"

        def unpack_legacy(self, data_item: DataItem) -> str:
            return str(data_item.key)

    packager = UnpackOnlyPackager()

    # Unpacking artifact types should include "legacy":
    assert "legacy" in packager.get_supported_unpacking_artifact_types()

    # Packing artifact types should NOT include "legacy":
    assert "legacy" not in packager.get_supported_packing_artifact_types()

    # Union (get_supported_artifact_types) should include "legacy":
    assert "legacy" in packager.get_supported_artifact_types()

    # is_packable should return False for artifact_type="legacy":
    assert packager.is_packable(obj="hello", artifact_type="legacy") is False

    # is_unpackable should return True for artifact_type="legacy":
    assert (
        packager.is_unpackable(
            data_item=DummyDataItem(key="test"),
            type_hint=str,
            artifact_type="legacy",
        )
        is True
    )
