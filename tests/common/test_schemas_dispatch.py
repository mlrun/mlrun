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

import importlib

import pytest

import mlrun.common.schemas as schemas
from mlrun.common.schemas._dispatch import USE_V2_SCHEMAS


def test_dispatcher_selects_face_by_environment():
    # the face is chosen by environment (USE_V2_SCHEMAS), not the installed Pydantic major
    expected_face = "_v2" if USE_V2_SCHEMAS else "_v1"
    assert (
        schemas.AlertConfig.__module__ == f"mlrun.common.schemas.{expected_face}.alert"
    )
    assert schemas.PatchMode.__module__ == "mlrun.common.schemas._shared.constants"


def test_v2_face_dormant_this_story():
    # the _v2 face is not enabled yet, so every environment (the API server too) binds _v1
    assert USE_V2_SCHEMAS is False
    assert schemas.AlertConfig.__module__ == "mlrun.common.schemas._v1.alert"


def test_v2_package_present_but_empty():
    from mlrun.common.schemas import _v2

    def public_names(module):
        return [name for name in vars(module) if not name.startswith("_")]

    # the package and per-topic stubs exist (so the facade dispatch resolves) but
    # define nothing yet
    assert public_names(_v2) == []
    assert public_names(importlib.import_module("mlrun.common.schemas._v2.alert")) == []
    assert (
        public_names(
            importlib.import_module(
                "mlrun.common.schemas._v2.model_monitoring.model_endpoints"
            )
        )
        == []
    )


@pytest.mark.parametrize(
    "name",
    [
        "LogCollectorEventActions",
        "AlertConfig",
        "PatchMode",
        "AuthInfo",
        "ModelEndpoint",
        "TSDBTarget",
    ],
)
def test_flat_public_name_importable(name):
    assert hasattr(schemas, name), f"mlrun.common.schemas.{name} must stay importable"


@pytest.mark.parametrize(
    "module_path,name",
    [
        # names a base submodule only imports (leaked), but callers reach through the
        # submodule path, so the facade must keep re-exporting them
        ("mlrun.common.schemas.schedule", "LabelRecord"),
        ("mlrun.common.schemas.feature_store", "ObjectMetadata"),
        ("mlrun.common.schemas.feature_store", "ObjectStatus"),
        ("mlrun.common.schemas.alert", "AlertConfig"),
        ("mlrun.common.schemas.constants", "PatchMode"),
        ("mlrun.common.schemas.model_monitoring.constants", "TSDBTarget"),
        ("mlrun.common.schemas.model_monitoring.model_endpoints", "ModelEndpoint"),
        (
            "mlrun.common.schemas.model_monitoring.model_endpoints",
            "parse_metric_fqn_to_monitoring_metric",
        ),
        ("mlrun.common.schemas.model_monitoring.model_endpoints", "compose_full_name"),
    ],
)
def test_submodule_public_name_importable(module_path, name):
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path}.{name} must stay importable"


def test_event_kind_entity_map_lives_in_shared():
    # version-agnostic map: single source in _shared, not leaked onto the public facade
    from mlrun.common.schemas._shared.alert import _event_kind_entity_map  # noqa: F401

    facade = importlib.import_module("mlrun.common.schemas.alert")
    assert not hasattr(facade, "_event_kind_entity_map")


@pytest.mark.parametrize(
    "name,submodule",
    [
        ("AlertConfig", "alert"),
        ("PatchMode", "constants"),
        ("AuthInfo", "auth"),
        ("FeatureSet", "feature_store"),
    ],
)
def test_flat_name_is_same_object_as_submodule(name, submodule):
    module = importlib.import_module(f"mlrun.common.schemas.{submodule}")
    assert getattr(schemas, name) is getattr(module, name)
