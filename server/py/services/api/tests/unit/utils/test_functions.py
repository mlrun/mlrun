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

from unittest.mock import MagicMock, patch

import pytest

import mlrun
import mlrun.errors

from services.api.utils.functions import enrich_function_from_code_artifact


def _mock_code_artifact(requirements=None):
    """Build a MagicMock standing in for a CodeArtifact with kind='code'."""
    artifact = MagicMock()
    artifact.kind = "code"
    artifact.spec.requirements = requirements
    return artifact


def test_enrich_function_user_requirements_take_priority():
    """User requirements win over artifact requirements for same package."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    func.spec.build.requirements = ["pandas>=1.5"]

    artifact = _mock_code_artifact(requirements=["pandas>=2.0"])

    with patch("mlrun.datastore.get_store_resource", return_value=artifact):
        enrich_function_from_code_artifact(func, "proj")

    pandas_reqs = [r for r in func.spec.build.requirements if "pandas" in r.lower()]
    assert len(pandas_reqs) == 1
    assert "1.5" in pandas_reqs[0]


def test_enrich_function_no_requirements_is_noop():
    """Artifact with no requirements does not change function requirements."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    func.spec.build.requirements = ["existing-pkg"]

    artifact = _mock_code_artifact(requirements=None)

    with patch("mlrun.datastore.get_store_resource", return_value=artifact):
        enrich_function_from_code_artifact(func, "proj")

    assert func.spec.build.requirements == ["existing-pkg"]


def test_enrich_function_non_store_source_is_noop():
    """Non-store:// source skips artifact resolution entirely."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "/local/path/code.py"

    with patch("mlrun.datastore.get_store_resource") as mock_get:
        enrich_function_from_code_artifact(func, "proj")
        mock_get.assert_not_called()


def test_enrich_function_unexpected_failure_wraps_as_invalid_argument():
    """Unexpected (non-MLRun) errors are wrapped as MLRunInvalidArgumentError."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/missing"

    with patch(
        "mlrun.datastore.get_store_resource",
        side_effect=RuntimeError("connection reset"),
    ):
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Cannot resolve code artifact",
        ):
            enrich_function_from_code_artifact(func, "proj")


def test_enrich_function_preserves_typed_mlrun_errors():
    """Typed MLRun errors (e.g. NotFound → 404) pass through unwrapped."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/missing"

    with patch(
        "mlrun.datastore.get_store_resource",
        side_effect=mlrun.errors.MLRunNotFoundError("artifact not found"),
    ):
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            enrich_function_from_code_artifact(func, "proj")


def test_enrich_function_rejects_non_code_artifact_kind():
    """Resolved artifact must be a CodeArtifact (kind='code'); other kinds raise."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/some_model"

    artifact = MagicMock()
    artifact.kind = "model"

    with patch("mlrun.datastore.get_store_resource", return_value=artifact):
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="resolves to a 'model' artifact",
        ):
            enrich_function_from_code_artifact(func, "proj")


def test_enrich_function_defaults_load_source_on_run_to_true():
    """Unset load_source_on_run defaults to True for store:// sources."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    # load_source_on_run left as default (None)

    artifact = _mock_code_artifact(requirements=None)

    with patch("mlrun.datastore.get_store_resource", return_value=artifact):
        enrich_function_from_code_artifact(func, "proj")

    assert func.spec.build.load_source_on_run is True


def test_enrich_function_preserves_explicit_load_source_on_run_false():
    """Explicit load_source_on_run=False is preserved for store:// sources."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    func.spec.build.load_source_on_run = False  # explicit user value

    artifact = _mock_code_artifact(requirements=None)

    with patch("mlrun.datastore.get_store_resource", return_value=artifact):
        enrich_function_from_code_artifact(func, "proj")

    assert func.spec.build.load_source_on_run is False


def test_enrich_function_falls_back_to_application_source():
    """When spec.build.source is empty, falls back to status.application_source.

    Scaffolding for ML-12480 (Nuclio dual-mode): ApplicationRuntime.from_image()
    stashes the original spec.build.source into status.application_source, so
    subsequent enrich passes need to resolve from there. Inert for runtimes
    that don't set application_source.
    """
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = ""
    func.status.application_source = "store://artifacts/proj/my_code"

    artifact = _mock_code_artifact(requirements=["pandas"])

    with patch("mlrun.datastore.get_store_resource", return_value=artifact) as mock_get:
        enrich_function_from_code_artifact(func, "proj")

    mock_get.assert_called_once()
    assert "pandas" in func.spec.build.requirements
