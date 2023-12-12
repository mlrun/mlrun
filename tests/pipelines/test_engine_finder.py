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

import pytest

import mlrun.pipelines


@pytest.mark.parametrize(
    "fullname, path, final_import_path",
    [
        (
            "mlrun.pipelines.iguazio",
            ["/app/iguazio/mlrun/mlrun/pipelines"],
            "/app/iguazio/mlrun/mlrun/pipelines/kfp/v1_8/iguazio.py",
        ),
        (
            "mlrun.pipelines.ops",
            ["/app/iguazio/mlrun/mlrun/pipelines"],
            "/app/iguazio/mlrun/mlrun/pipelines/kfp/v1_8/ops.py",
        ),
        (
            "mlrun.pipelines.utils",
            ["/app/iguazio/mlrun/mlrun/pipelines"],
            "/app/iguazio/mlrun/mlrun/pipelines/kfp/v1_8/utils.py",
        ),
        (
            "mlrun.pipelines.api.utils",
            ["/app/iguazio/mlrun/mlrun/pipelines"],
            "/app/iguazio/mlrun/mlrun/pipelines/kfp/v1_8/api/utils.py",
        ),
    ],
)
def test_pipeline_engine_path_finder(
    fullname: str, path: str, final_import_path: str, monkeypatch
):
    monkeypatch.setattr(mlrun.pipelines, "PIPELINE_COMPATIBILITY_MODE", "kfp-v1.8")
    assert (
        mlrun.pipelines.PipelineEngineModuleFinder._resolve_module_path(fullname, path)
        == final_import_path
    )
