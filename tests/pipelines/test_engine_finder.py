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
