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

import mlrun
import mlrun.errors

from services.api.utils.functions import enrich_function_from_code_artifact


def test_enrich_function_merges_artifact_requirements():
    """Artifact requirements are merged into function build spec."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    func.spec.build.requirements = []

    mock_artifact = MagicMock()
    mock_artifact.spec.requirements = ["pandas>=2.0", "numpy"]

    with patch("mlrun.datastore.get_store_resource", return_value=mock_artifact):
        enrich_function_from_code_artifact(func, "proj")

    assert "pandas>=2.0" in func.spec.build.requirements
    assert "numpy" in func.spec.build.requirements


def test_enrich_function_user_requirements_take_priority():
    """User requirements win over artifact requirements for same package."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    func.spec.build.requirements = ["pandas>=1.5"]

    mock_artifact = MagicMock()
    mock_artifact.spec.requirements = ["pandas>=2.0"]

    with patch("mlrun.datastore.get_store_resource", return_value=mock_artifact):
        enrich_function_from_code_artifact(func, "proj")

    pandas_reqs = [r for r in func.spec.build.requirements if "pandas" in r.lower()]
    assert len(pandas_reqs) == 1
    assert "1.5" in pandas_reqs[0]


def test_enrich_function_no_requirements_is_noop():
    """Artifact with no requirements does not change function."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/my_code"
    func.spec.build.requirements = ["existing-pkg"]

    mock_artifact = MagicMock()
    mock_artifact.spec.requirements = None

    with patch("mlrun.datastore.get_store_resource", return_value=mock_artifact):
        enrich_function_from_code_artifact(func, "proj")

    assert func.spec.build.requirements == ["existing-pkg"]


def test_enrich_function_non_store_source_is_noop():
    """Non-store:// source skips artifact resolution entirely."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "/local/path/code.py"

    with patch("mlrun.datastore.get_store_resource") as mock_get:
        enrich_function_from_code_artifact(func, "proj")
        mock_get.assert_not_called()


def test_enrich_function_artifact_resolution_failure_raises():
    """Artifact resolution failure raises clear error."""
    func = mlrun.new_function("test", kind="job")
    func.spec.build.source = "store://artifacts/proj/missing"

    with patch(
        "mlrun.datastore.get_store_resource",
        side_effect=Exception("artifact not found"),
    ):
        try:
            enrich_function_from_code_artifact(func, "proj")
            assert False, "Should have raised"
        except mlrun.errors.MLRunInvalidArgumentError as exc:
            assert "Cannot resolve code artifact" in str(exc)
