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
import os

import pydantic
import pytest

import mlrun.common.schemas as schemas
from mlrun.common.schemas._dispatch import USE_V2_SCHEMAS

# The native ``_v2`` modules use pydantic-2-only APIs (e.g. ``conlist(min_length=...)``) and are
# only importable on the pydantic-2 baseline — the environment the API server actually runs in.
requires_pydantic2 = pytest.mark.skipif(
    int(pydantic.VERSION.split(".")[0]) < 2,
    reason="native _v2 schemas require the pydantic 2 baseline",
)


@pytest.fixture
def rebind_face():
    """Re-evaluate the environment-keyed dispatch under a chosen ``MLRUN_IS_API_SERVER`` value
    and restore the original binding afterwards. Reloads ``_dispatch`` plus the ``secret`` facade
    (a representative dispatched module) so the selected face is observable without a subprocess."""
    import mlrun.common.schemas._dispatch as dispatch
    import mlrun.common.schemas.secret as secret_facade

    original = os.environ.get("MLRUN_IS_API_SERVER")

    def _rebind(is_api_server: bool):
        os.environ["MLRUN_IS_API_SERVER"] = "true" if is_api_server else "false"
        importlib.reload(dispatch)
        importlib.reload(secret_facade)
        return dispatch, secret_facade

    yield _rebind

    if original is None:
        os.environ.pop("MLRUN_IS_API_SERVER", None)
    else:
        os.environ["MLRUN_IS_API_SERVER"] = original
    importlib.reload(dispatch)
    importlib.reload(secret_facade)


def test_dispatcher_selects_face_by_environment():
    # the face is chosen by environment (USE_V2_SCHEMAS), not the installed Pydantic major
    expected_face = "_v2" if USE_V2_SCHEMAS else "_v1"
    assert (
        schemas.AlertConfig.__module__ == f"mlrun.common.schemas.{expected_face}.alert"
    )
    assert schemas.PatchMode.__module__ == "mlrun.common.schemas._shared.constants"


def test_dispatch_flag_follows_api_server_env():
    # the flip (ML-12891/12892) binds the face purely to whether this is the API server process
    from mlrun.common.schemas._dispatch import IS_API_SERVER

    assert USE_V2_SCHEMAS == IS_API_SERVER


@requires_pydantic2
def test_api_server_binds_v2_face(rebind_face):
    dispatch, secret_facade = rebind_face(is_api_server=True)
    assert dispatch.USE_V2_SCHEMAS is True
    assert secret_facade.SecretsData.__module__ == "mlrun.common.schemas._v2.secret"


def test_non_api_server_binds_v1_face(rebind_face):
    dispatch, secret_facade = rebind_face(is_api_server=False)
    assert dispatch.USE_V2_SCHEMAS is False
    assert secret_facade.SecretsData.__module__ == "mlrun.common.schemas._v1.secret"


@requires_pydantic2
def test_v2_face_is_populated():
    # ML-12891 filled the native v2 face — the inverse of the pre-flip "empty stub" invariant.
    # Full structural parity with _v1 is asserted in test_schemas_parity.
    def public_names(module):
        return [name for name in vars(module) if not name.startswith("_")]

    # the per-topic modules now define the native v2 models (aggregation still lives in the
    # top-level facade, not the _v2 package __init__)
    assert "AlertConfig" in public_names(
        importlib.import_module("mlrun.common.schemas._v2.alert")
    )
    assert "ModelEndpoint" in public_names(
        importlib.import_module(
            "mlrun.common.schemas._v2.model_monitoring.model_endpoints"
        )
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
