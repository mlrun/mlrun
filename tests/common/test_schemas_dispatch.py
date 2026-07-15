# Copyright 2024 Iguazio
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
"""Scaffolding guards for the version-dispatched ``mlrun.common.schemas`` package
(ML-12890). These lock the public-import-path contract the split must preserve:
the flat names, the per-topic submodule paths, and the private helpers callers
import — plus the dispatcher and the empty ``_v2`` invariants.
"""

import importlib

import pytest

import mlrun.common.schemas as schemas
from mlrun.common.schemas._dispatch import USE_V2_SCHEMAS


def test_dispatcher_selects_face_by_environment():
    """The model face is chosen by *environment* (``USE_V2_SCHEMAS``), not by the
    installed Pydantic major; shared definitions always resolve from ``_shared``."""
    expected_face = "_v2" if USE_V2_SCHEMAS else "_v1"
    assert (
        schemas.AlertConfig.__module__ == f"mlrun.common.schemas.{expected_face}.alert"
    )
    assert schemas.PatchMode.__module__ == "mlrun.common.schemas._shared.constants"


def test_v2_face_dormant_this_story():
    """ML-12890 keeps the split dormant: the ``_v2`` face is not enabled yet, so every
    environment (the API server included) binds ``_v1``. ML-12892 flips this on."""
    assert USE_V2_SCHEMAS is False
    assert schemas.AlertConfig.__module__ == "mlrun.common.schemas._v1.alert"


def test_v2_package_present_but_empty():
    """ML-12890 delivers ``_v2`` as an empty face; native v2 models arrive in ML-12891."""
    from mlrun.common.schemas import _v2

    assert _v2.__all__ == []
    # per-topic stubs exist (so the facade dispatch resolves) but export nothing yet
    assert importlib.import_module("mlrun.common.schemas._v2.alert").__all__ == []
    assert (
        importlib.import_module(
            "mlrun.common.schemas._v2.model_monitoring.model_endpoints"
        ).__all__
        == []
    )


@pytest.mark.parametrize(
    "name",
    [
        # regression: this flat export was dropped when the events enums moved to
        # ``_shared`` but the dispatcher's re-export list omitted it.
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
        # regressions: names a base submodule only *imported* (leaked) but that
        # callers reference through the submodule path.
        ("mlrun.common.schemas.schedule", "LabelRecord"),
        ("mlrun.common.schemas.feature_store", "ObjectMetadata"),
        ("mlrun.common.schemas.feature_store", "ObjectStatus"),
        # ordinary defined names on their own topic
        ("mlrun.common.schemas.alert", "AlertConfig"),
        ("mlrun.common.schemas.constants", "PatchMode"),
        ("mlrun.common.schemas.model_monitoring.constants", "TSDBTarget"),
        ("mlrun.common.schemas.model_monitoring.model_endpoints", "ModelEndpoint"),
    ],
)
def test_submodule_public_name_importable(module_path, name):
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path}.{name} must stay importable"


@pytest.mark.parametrize(
    "module_path,name",
    [
        # regressions: private helpers imported directly by callers/tests; ``import *``
        # would drop them, so the facades mirror the full module namespace.
        ("mlrun.common.schemas.alert", "_event_kind_entity_map"),
        (
            "mlrun.common.schemas.model_monitoring.model_endpoints",
            "_parse_metric_fqn_to_monitoring_metric",
        ),
    ],
)
def test_private_helper_still_importable(module_path, name):
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path}.{name} must stay importable"


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
    """The flat re-export and the per-topic submodule expose the *same* object, so a
    consumer using either path gets identical behaviour."""
    module = importlib.import_module(f"mlrun.common.schemas.{submodule}")
    assert getattr(schemas, name) is getattr(module, name)
