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

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

import mlrun
import mlrun.render
from mlrun.lists import RunList
from tests.conftest import results, rundb_path

assets_path = pathlib.Path(__file__).parent / "assets"
function_path = str(assets_path / "log_function.py")


def get_db():
    return mlrun.get_run_db(rundb_path)


@pytest.mark.parametrize(
    "generate_artifact_hash_mode, expected_target_paths",
    [
        (
            False,
            [
                f"{results}/log-function-log-dataset/0/feature_1.csv",
                f"{results}/log-function-log-dataset/0/feature_2.csv",
            ],
        ),
        (
            True,
            [
                f"{results}/6154c46f1a6fffb0b6b716882279d7e09ecb6b8a.csv",
                f"{results}/c88c2dc877a6595cb2eb834449aac6e2789d301c.csv",
            ],
        ),
    ],
)
def test_list_runs(rundb_mock, generate_artifact_hash_mode, expected_target_paths):
    mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = (
        generate_artifact_hash_mode
    )
    mlrun.mlconf.ui.url = "http://mlrun-ui:8080"
    func = mlrun.code_to_function(
        filename=function_path, kind="job", handler="log_dataset"
    )
    run = func.run(local=True, output_path=str(results))

    # Verify target path in enriched run list
    runs = RunList([run.to_dict()])
    html = runs.show(display=False)
    assert (
        f"{mlrun.mlconf.ui.url}/"
        f"projects/"
        f"{run.metadata.project}/"
        f"jobs/monitor-jobs/"
        f"{run.metadata.name}/"
        f"{run.metadata.uid}/"
        f"overview" in html
    )
    for expected_target_path in expected_target_paths:
        expected_link, _ = mlrun.render.link_to_ipython(expected_target_path)
        assert expected_link in html

    runs = rundb_mock.list_runs()
    assert runs, "empty runs result"

    # Verify store URI in not-enriched runs
    html = runs.show(display=False)
    dataset_0_uri = list(runs[0]["status"]["artifact_uris"].values())[0]
    assert dataset_0_uri
    assert dataset_0_uri in html


DELETE_OUTPUTS = True  # Set to False to keep HTML files for visual inspection.
OUTPUT_DIR = pathlib.Path(__file__).parent / "render_test_outputs"


def consume_nested_inputs(
    datasets: dict[str, mlrun.DataItem],
    file_list: list,
) -> tuple[int, int]:
    """
    Handler that receives bundled (dict/list) inputs.
    """
    return len(datasets), len(file_list)


def train_and_predict(
    learning_rate: float,
    num_epochs: int,
    training_data: mlrun.DataItem,
) -> tuple[pd.DataFrame, str, float, float]:
    """
    Returns predictions, model name, accuracy, and loss.
    """
    _df = training_data.as_df()
    predictions_df = pd.DataFrame(
        {
            "prediction": [0.9, 0.1, 0.8, 0.2, 0.7],
            "actual": [1, 0, 1, 0, 1],
        }
    )
    return predictions_df, "my-awesome-model-v1", 0.95, 0.05


def grade_records(
    record_list: list[mlrun.DataItem],
) -> tuple[list[dict], int]:
    """
    List input, list-of-dicts output.
    """
    return [
        {"name": "alice", "score": 91, "grade": "A"},
        {"name": "bob", "score": 82, "grade": "B"},
        {"name": "charlie", "score": 73, "grade": "C"},
    ], len(record_list)


def predict_from_splits(
    data_splits: dict[str, pd.DataFrame],
) -> tuple[list[pd.DataFrame], int, int, float]:
    """
    Returns prediction DataFrames, total rows, split count, and a mean value.
    """
    split_names = list(data_splits.keys())
    total_rows = sum(data_splits[name].shape[0] for name in split_names)
    return (
        [
            pd.DataFrame({"predicted": [0.9, 0.1, 0.8], "actual": [1, 0, 1]}),
            pd.DataFrame({"predicted": [0.2, 0.7], "actual": [0, 1]}),
        ],
        int(total_rows),
        len(split_names),
        float(np.mean([1.0, 2.0, 3.0])),
    )


def produce_dataset_bundle() -> tuple[dict[str, pd.DataFrame], float]:
    """
    Return a dict of DataFrames as a bundled output, plus a scalar metric.
    """
    return {
        "train_data": pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        "test_data": pd.DataFrame({"x": [7, 8], "y": [9, 10]}),
        "val_data": pd.DataFrame({"x": [11], "y": [12]}),
    }, 0.95


def _write_test_html(test_name: str, table_html: str):
    """
    Write a standalone HTML file for visual inspection of a render test only if DELETE_OUTPUTS is False.

    :param test_name:  name of the test.
    :param table_html: the HTML string of the rendered table to include in the output.
    """
    if DELETE_OUTPUTS:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{test_name}</title>
    {mlrun.render.get_style()}
</head>
<body>
    <h1>{test_name}</h1>
    <p>Generated by <code>mlrun/tests/test_render.py::{test_name}</code></p>
    <hr>
    {mlrun.render.jscripts}
    {mlrun.render.tblframe.format(table_html)}
</body>
</html>"""

    output_path = OUTPUT_DIR / f"{test_name}.html"
    output_path.write_text(full_html, encoding="utf-8")
    print(f"\nHTML output written to: {output_path}")


def test_inputs_results_artifacts(rundb_mock):
    """
    Run a handler with scalar params, a dataset input, and mixed result/artifact returns.
    """
    artifact_path = tempfile.TemporaryDirectory()

    # Create project:
    project = mlrun.get_or_create_project("render-test", allow_cross_project=True)

    # Set function from this file:
    fn = project.set_function(func=__file__, name="test-render", kind="job")

    # Log a training dataset directly via the project:
    training_artifact = project.log_dataset(
        "training_data",
        df=pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [10, 20, 30, 40, 50],
                "label": [0, 1, 0, 1, 0],
            }
        ),
        artifact_path=artifact_path.name,
        format="csv",
    )
    training_data_uri = training_artifact.uri

    # Run train_and_predict with params, inputs, and returns log hints:
    run = fn.run(
        handler="train_and_predict",
        params={"learning_rate": 0.001, "num_epochs": 10},
        inputs={"training_data": training_data_uri},
        returns=[
            "predictions",
            "model_name: result",
            "accuracy: result",
            "loss: result",
        ],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Build RunList and render HTML (display=False -> returns raw HTML string):
    runs = RunList([run.to_dict()])
    table_html = runs.show(display=False)
    assert table_html is not None, "RunList.show(display=False) returned None"

    # Assert key content appears in the rendered HTML:
    # Results
    assert "accuracy" in table_html, "Missing result: accuracy"
    assert "loss" in table_html, "Missing result: loss"
    assert "model_name" in table_html, "Missing result: model_name"

    # Parameters
    assert "learning_rate" in table_html, "Missing parameter: learning_rate"
    assert "num_epochs" in table_html, "Missing parameter: num_epochs"

    # Inputs
    assert "training_data" in table_html, "Missing input: training_data"

    # Artifacts
    assert "predictions" in table_html, "Missing artifact: predictions"

    # Write HTML output:
    _write_test_html(
        test_name="test_inputs_results_artifacts",
        table_html=table_html,
    )

    # Cleanup:
    artifact_path.cleanup()


def test_nested_inputs(rundb_mock):
    """
    Run a handler that receives dict-bundled and list-bundled dataset inputs.
    """
    artifact_path = tempfile.TemporaryDirectory()

    # Create project:
    project = mlrun.get_or_create_project(
        "render-bundle-test", allow_cross_project=True
    )

    # Set function from this file:
    fn = project.set_function(func=__file__, name="test-bundle-render", kind="job")

    # Log multiple datasets directly via the project:
    small_df = pd.DataFrame({"col": [1, 2, 3]})
    uri_a = project.log_dataset(
        "dataset_a", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri
    uri_b = project.log_dataset(
        "dataset_b", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri
    uri_c = project.log_dataset(
        "dataset_c", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri

    # Run with bundled inputs (dict + list):
    run = fn.run(
        handler="consume_nested_inputs",
        inputs={
            "datasets": {"train": uri_a, "test": uri_b},
            "file_list": [uri_a, uri_b, uri_c],
        },
        returns=["num_datasets: result", "num_files: result"],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Build RunList and render HTML — this crashed before the fix:
    runs = RunList([run.to_dict()])
    table_html = runs.show(display=False)
    assert table_html is not None, "RunList.show(display=False) returned None"

    # Assert bundled inputs appear in the rendered HTML:
    # Dict bundle children
    assert "datasets" in table_html, "Missing bundle key: datasets"
    assert "train" in table_html, "Missing dict child key: train"
    assert "test" in table_html, "Missing dict child key: test"
    # List bundle children
    assert "file_list" in table_html, "Missing bundle key: file_list"
    assert "[0]" in table_html, "Missing list child: [0]"
    assert "[1]" in table_html, "Missing list child: [1]"
    assert "[2]" in table_html, "Missing list child: [2]"
    # CSS classes
    assert "input-bundle" in table_html, "Missing CSS class: input-bundle"
    assert "input-bundle-key" in table_html, "Missing CSS class: input-bundle-key"

    # Write HTML output:
    _write_test_html(
        test_name="test_nested_inputs",
        table_html=table_html,
    )

    # Cleanup:
    artifact_path.cleanup()


def test_nested_results(rundb_mock):
    """
    Run a handler that returns a dict-bundled dataset output and a scalar result.
    """
    artifact_path = tempfile.TemporaryDirectory()

    # Create project:
    project = mlrun.get_or_create_project(
        "render-output-test", allow_cross_project=True
    )

    # Set function from this file:
    fn = project.set_function(func=__file__, name="test-output-render", kind="job")

    # Run handler that returns a dict of DataFrames with bundle log hint:
    run = fn.run(
        handler="produce_dataset_bundle",
        returns=["*my_datasets", "scalar_metric: result"],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Build RunList and render HTML:
    runs = RunList([run.to_dict()])
    table_html = runs.show(display=False)
    assert table_html is not None, "RunList.show(display=False) returned None"

    # Assert bundled output results appear grouped in the HTML:
    # The bundle key should appear
    assert "my_datasets" in table_html, "Missing bundle key: my_datasets"
    # Child keys from the bundle structure
    assert "train_data" in table_html, "Missing child key: train_data"
    assert "test_data" in table_html, "Missing child key: test_data"
    assert "val_data" in table_html, "Missing child key: val_data"
    # Scalar result should still appear
    assert "scalar_metric" in table_html, "Missing scalar result: scalar_metric"
    # CSS classes for grouped rendering
    assert "input-bundle" in table_html, "Missing CSS class: input-bundle"
    assert "input-bundle-key" in table_html, "Missing CSS class: input-bundle-key"

    # Write HTML output:
    _write_test_html(
        test_name="test_nested_results",
        table_html=table_html,
    )

    # Cleanup:
    artifact_path.cleanup()


def test_nested_inputs_nested_results(rundb_mock):
    """
    Run a handler with list-bundled inputs that returns a list-of-dicts bundle and a scalar result.
    """
    artifact_path = tempfile.TemporaryDirectory()

    # Create project:
    project = mlrun.get_or_create_project(
        "render-complex-io-test", allow_cross_project=True
    )

    # Set function from this file:
    fn = project.set_function(func=__file__, name="test-complex-io", kind="job")

    # Log multiple datasets directly via the project:
    small_df = pd.DataFrame({"col": [1, 2, 3]})
    uri_a = project.log_dataset(
        "dataset_a", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri
    uri_b = project.log_dataset(
        "dataset_b", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri
    uri_c = project.log_dataset(
        "dataset_c", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri

    # Run with bundled list input and list-of-dicts output:
    run = fn.run(
        handler="grade_records",
        inputs={"record_list": [uri_a, uri_b, uri_c]},
        returns=["*student_records", "num_records_in: result"],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Build RunList and render HTML:
    runs = RunList([run.to_dict()])
    table_html = runs.show(display=False)
    assert table_html is not None, "RunList.show(display=False) returned None"

    # Assert bundled list inputs appear:
    assert "record_list" in table_html, "Missing input bundle key: record_list"
    assert "[0]" in table_html, "Missing list child: [0]"
    assert "[1]" in table_html, "Missing list child: [1]"
    assert "[2]" in table_html, "Missing list child: [2]"

    # Bundled list-of-dicts output results
    assert "student_records" in table_html, "Missing output bundle key: student_records"

    # Dict keys inside each list element should be rendered (not just [0], [1], [2]):
    assert "name=" in table_html, "Missing dict key in list element: name"
    assert "score=" in table_html, "Missing dict key in list element: score"
    assert "grade=" in table_html, "Missing dict key in list element: grade"

    # Scalar result
    assert "num_records_in" in table_html, "Missing scalar result: num_records_in"

    # Individual artifacts from unbundled list
    assert "student_records_0" in table_html, "Missing artifact: student_records_0"
    assert "student_records_1" in table_html, "Missing artifact: student_records_1"
    assert "student_records_2" in table_html, "Missing artifact: student_records_2"

    # CSS classes for bundle rendering
    assert "input-bundle" in table_html, "Missing CSS class: input-bundle"
    assert "input-bundle-key" in table_html, "Missing CSS class: input-bundle-key"

    # Write HTML output:
    _write_test_html(
        test_name="test_nested_inputs_nested_results",
        table_html=table_html,
    )

    # Cleanup:
    artifact_path.cleanup()


def test_nested_inputs_nested_artifacts(rundb_mock):
    """
    Run a handler with dict-bundled DataFrame inputs returning list-bundled artifacts and scalars.
    """
    artifact_path = tempfile.TemporaryDirectory()

    # Create project:
    project = mlrun.get_or_create_project("render-df-io-test", allow_cross_project=True)

    # Set function from this file:
    fn = project.set_function(func=__file__, name="test-df-io", kind="job")

    # Log two datasets directly via the project:
    small_df = pd.DataFrame({"col": [1, 2, 3]})
    uri_a = project.log_dataset(
        "dataset_a", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri
    uri_b = project.log_dataset(
        "dataset_b", df=small_df, artifact_path=artifact_path.name, format="csv"
    ).uri

    # Run with dict-of-DataFrames bundled input and list-of-DataFrames output:
    run = fn.run(
        handler="predict_from_splits",
        inputs={"data_splits": {"train": uri_a, "test": uri_b}},
        returns=[
            "*prediction_sets",
            "total_rows: result",
            "num_splits: result",
            "mean_value: result",
        ],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Build RunList and render HTML:
    runs = RunList([run.to_dict()])
    table_html = runs.show(display=False)
    assert table_html is not None, "RunList.show(display=False) returned None"

    # Assert input bundle structure:
    assert "data_splits" in table_html, "Missing input bundle key: data_splits"
    assert "train" in table_html, "Missing dict child key: train"
    assert "test" in table_html, "Missing dict child key: test"

    # Output bundle structure
    assert "prediction_sets" in table_html, "Missing output bundle key: prediction_sets"
    assert "[0]" in table_html, "Missing list child: [0]"
    assert "[1]" in table_html, "Missing list child: [1]"

    # Scalar results from numpy computations
    assert "total_rows" in table_html, "Missing scalar result: total_rows"
    assert "num_splits" in table_html, "Missing scalar result: num_splits"
    assert "mean_value" in table_html, "Missing scalar result: mean_value"

    # Individual artifacts from unbundled list output
    assert "prediction_sets_0" in table_html, "Missing artifact: prediction_sets_0"
    assert "prediction_sets_1" in table_html, "Missing artifact: prediction_sets_1"

    # CSS classes for bundle rendering
    assert "input-bundle" in table_html, "Missing CSS class: input-bundle"
    assert "input-bundle-key" in table_html, "Missing CSS class: input-bundle-key"

    # Write HTML output:
    _write_test_html(
        test_name="test_nested_inputs_nested_artifacts",
        table_html=table_html,
    )

    # Cleanup:
    artifact_path.cleanup()
