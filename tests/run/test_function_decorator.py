# Copyright 2018 Iguazio
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
import os
import tempfile
import zipfile
from typing import Tuple

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import mlrun


@mlrun.function(labels={"a": 1, "b": "a test", "c": [1, 2, 3]})
def set_labels(arg1, arg2=23):
    return arg1 - arg2


def test_set_labels_without_mlrun():
    """
    Run the `set_labels` function without MLRun to see the wrapper is transparent.
    """
    returned_result = set_labels(24)
    assert returned_result == 1

    returned_result = set_labels(20, 18)
    assert returned_result == 2

    returned_result = set_labels(arg1=24)
    assert returned_result == 1

    returned_result = set_labels(arg1=20, arg2=18)
    assert returned_result == 2


def test_set_labels_with_mlrun():
    """
    Run the `set_labels` function with MLRun to see the wrapper is setting the required labels.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="set_labels",
        params={"arg1": 24},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.metadata.labels)

    # Assertion:
    assert run_object.metadata.labels["a"] == "1"
    assert run_object.metadata.labels["b"] == "a test"
    assert run_object.metadata.labels["c"] == "[1, 2, 3]"

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(labels={"wrapper_label": "2"})
def set_labels_from_function_and_wrapper(context: mlrun.MLClientCtx = None):
    if context:
        context.set_label("context_label", 1)


def test_set_labels_from_function_and_wrapper_without_mlrun():
    """
    Run the `set_labels_from_function_and_wrapper` function without MLRun to see the wrapper is transparent.
    """
    returned_result = set_labels_from_function_and_wrapper()
    assert returned_result is None


def test_set_labels_from_function_and_wrapper_with_mlrun():
    """
    Run the `set_labels_from_function_and_wrapper` function with MLRun to see the wrapper is setting the required
    labels without interrupting to the ones set via the context by the user.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="set_labels_from_function_and_wrapper",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.metadata.labels)

    # Assertion:
    assert run_object.metadata.labels["context_label"] == "1"
    assert run_object.metadata.labels["wrapper_label"] == "2"

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        ("my_array", mlrun.ArtifactType.DATASET),
        "my_df:dataset",
        ("my_dict", mlrun.ArtifactType.DATASET),
        ("my_list", "dataset"),
    ]
)
def log_dataset() -> Tuple[np.ndarray, pd.DataFrame, dict, list]:
    return (
        np.ones((10, 20)),
        pd.DataFrame(np.zeros((20, 10))),
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        [["A"], ["B"], [""]],
    )


def test_log_dataset_without_mlrun():
    """
    Run the `log_dataset` function without MLRun to see the wrapper is transparent.
    """
    my_array, my_df, my_dict, my_list = log_dataset()
    assert isinstance(my_array, np.ndarray)
    assert isinstance(my_df, pd.DataFrame)
    assert isinstance(my_dict, dict)
    assert isinstance(my_list, list)


def test_log_dataset_with_mlrun():
    """
    Run the `log_dataset` function with MLRun to see the wrapper is logging the returned values as datasets artifacts.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_dataset",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert (
        len(run_object.outputs) == 4
    )  # return + my_array, my_df, my_dict, my_list
    assert run_object.artifact("my_array").as_df().shape == (10, 20)
    assert run_object.artifact("my_df").as_df().shape == (20, 10)
    assert run_object.artifact("my_dict").as_df().shape == (4, 2)
    assert run_object.artifact("my_list").as_df().shape == (3, 1)

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        ("my_dir", mlrun.ArtifactType.DIRECTORY),
    ]
)
def log_directory(path: str) -> str:
    path = os.path.join(path, "my_new_dir")
    os.makedirs(path)
    open(os.path.join(path, "a.txt"), "a").close()
    open(os.path.join(path, "b.txt"), "a").close()
    open(os.path.join(path, "c.txt"), "a").close()
    return path


def test_log_directory_without_mlrun():
    """
    Run the `log_directory` function without MLRun to see the wrapper is transparent.
    """
    temp_dir = tempfile.TemporaryDirectory()
    my_dir = log_directory(temp_dir.name)
    assert isinstance(my_dir, str)
    temp_dir.cleanup()


def test_log_directory_with_mlrun():
    """
    Run the `log_directory` function with MLRun to see the wrapper is logging the directory as a zip file.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_directory",
        params={"path": artifact_path.name},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert len(run_object.outputs) == 1  # return + my_dir
    my_dir_zip = run_object.artifact("my_dir").local()
    my_dir = os.path.join(artifact_path.name, "extract_here")
    with zipfile.ZipFile(my_dir_zip, "r") as zip_ref:
        zip_ref.extractall(my_dir)
    my_dir_contents = os.listdir(my_dir)
    assert len(my_dir_contents) == 3
    assert "a.txt" in my_dir_contents
    assert "b.txt" in my_dir_contents
    assert "c.txt" in my_dir_contents

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        ("my_file", mlrun.ArtifactType.FILE),
    ]
)
def log_file(path: str) -> str:
    my_file = os.path.join(path, "a.txt")
    open(my_file, "a").close()
    return my_file


def test_log_file_without_mlrun():
    """
    Run the `log_file` function without MLRun to see the wrapper is transparent.
    """
    temp_dir = tempfile.TemporaryDirectory()
    my_file = log_file(temp_dir.name)
    assert isinstance(my_file, str)
    temp_dir.cleanup()


def test_log_file_with_mlrun():
    """
    Run the `log_file` function with MLRun to see the wrapper is logging the file.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_file",
        params={"path": artifact_path.name},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert len(run_object.outputs) == 1  # return + my_file
    assert os.path.basename(run_object.artifact("my_file").local()) == "my_file.txt"

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        ("my_object", mlrun.ArtifactType.OBJECT),
    ]
)
def log_object() -> Pipeline:
    encoder_to_imputer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(missing_values="", strategy="constant", fill_value="C"),
            ),
            ("encoder", OrdinalEncoder()),
        ]
    )
    encoder_to_imputer.fit([["A"], ["B"], ["C"]])
    return encoder_to_imputer


def test_log_object_without_mlrun():
    """
    Run the `log_object` function without MLRun to see the wrapper is transparent.
    """
    temp_dir = tempfile.TemporaryDirectory()
    my_object = log_object()
    assert isinstance(my_object, Pipeline)
    assert my_object.transform([["A"], ["B"], [""]]).tolist() == [[0], [1], [2]]
    temp_dir.cleanup()


def test_log_object_with_mlrun():
    """
    Run the `log_object` function with MLRun to see the wrapper is logging the object as pickle.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_object",
        params={"path": artifact_path.name},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert len(run_object.outputs) == 1  # return + my_file
    pickle = run_object.artifact("my_object").local()
    assert os.path.basename(pickle) == "my_object.pkl"
    with open(pickle, "rb") as pickle_file:
        my_object = cloudpickle.load(pickle_file)
    assert isinstance(my_object, Pipeline)
    assert my_object.transform([["A"], ["B"], [""]]).tolist() == [[0], [1], [2]]

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        ("my_plot", mlrun.ArtifactType.PLOT),
    ]
)
def log_plot() -> plt.Figure:
    my_plot, axes = plt.subplots()
    axes.plot([1, 2, 3, 4])
    return my_plot


def test_log_plot_without_mlrun():
    """
    Run the `log_plot` function without MLRun to see the wrapper is transparent.
    """
    my_plot = log_plot()
    assert isinstance(my_plot, plt.Figure)


def test_log_plot_with_mlrun():
    """
    Run the `log_plot` function with MLRun to see the wrapper is logging the plots as html files.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_plot",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert len(run_object.outputs) == 1  # return + my_plot
    assert os.path.basename(run_object.artifact("my_plot").local()) == "my_plot.html"

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        (
            "my_int",
            mlrun.ArtifactType.RESULT,
        ),
        "my_float",
        "my_dict: result",
        ("my_array",),
    ]
)
def log_result() -> Tuple[int, float, dict, np.ndarray]:
    return 1, 1.5, {"a": 1, "b": 2}, np.ones(3)


def test_log_result_without_mlrun():
    """
    Run the `log_result` function without MLRun to see the wrapper is transparent.
    """
    my_int, my_float, my_dict, my_array = log_result()
    assert isinstance(my_int, int)
    assert isinstance(my_float, float)
    assert isinstance(my_dict, dict)
    assert isinstance(my_array, np.ndarray)


def test_log_result_with_mlrun():
    """
    Run the `log_result` function with MLRun to see the wrapper is logging the returned values as results.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_result",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert (
        len(run_object.outputs) == 4
    )  # return + my_int, my_float, my_dict, my_array
    assert run_object.outputs["my_int"] == 1
    assert run_object.outputs["my_float"] == 1.5
    assert run_object.outputs["my_dict"] == {"a": 1, "b": 2}
    assert run_object.outputs["my_array"] == [1, 1, 1]

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(
    outputs=[
        ("wrapper_dataset", "dataset"),
        ("wrapper_result", mlrun.ArtifactType.RESULT),
    ]
)
def log_from_function_and_wrapper(context: mlrun.MLClientCtx = None):
    if context:
        context.log_result(key="context_result", value=1)
        context.log_dataset(key="context_dataset", df=pd.DataFrame(np.arange(10)))
    return [1, 2, 3, 4], "hello"


def test_log_from_function_and_wrapper_without_mlrun():
    """
    Run the `log_from_function_and_wrapper` function without MLRun to see the wrapper is transparent.
    """
    my_dataset, my_result = log_from_function_and_wrapper()
    assert isinstance(my_dataset, list)
    assert isinstance(my_result, str)


def test_log_from_function_and_wrapper_with_mlrun():
    """
    Run the `log_from_function_and_wrapper` function with MLRun to see the wrapper is logging the returned values
    among the other values logged via the context manually inside the function.
    """
    # Create the function and run:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    run_object = mlrun_function.run(
        handler="log_from_function_and_wrapper",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert (
        len(run_object.outputs) == 4
    )  # return + context_dataset, context_result, wrapper_dataset, wrapper_result
    assert run_object.artifact("context_dataset").as_df().shape == (10, 1)
    assert run_object.outputs["context_result"] == 1
    assert run_object.artifact("wrapper_dataset").as_df().shape == (4, 1)
    assert run_object.outputs["wrapper_result"] == "hello"

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function()
def parse_inputs_from_type_hints(
    my_data: list,
    my_encoder: Pipeline,
    another_data,
    add,
    mul: int = 2,
):
    assert another_data is None or isinstance(another_data, mlrun.DataItem)
    return (my_encoder.transform(my_data) + add * mul).tolist()


def test_parse_inputs_from_type_hints_without_mlrun():
    """
    Run the `parse_inputs_from_type_hints` function without MLRun to see the wrapper is transparent.
    """
    _, _, _, my_data = log_dataset()
    my_encoder = log_object()
    result = parse_inputs_from_type_hints(
        my_data, my_encoder=my_encoder, another_data=None, add=1
    )
    assert isinstance(result, list)
    assert result == [[2], [3], [4]]


def test_parse_inputs_from_type_hints_with_mlrun():
    """
    Run the `parse_inputs_from_type_hints` function with MLRun to see the wrapper is parsing the given inputs
    (`DataItem`s) to the written type hints.
    """
    # Create the function and run 2 of the previous functions to create a dataset and encoder objects:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    log_dataset_run = mlrun_function.run(
        handler="log_dataset",
        artifact_path=artifact_path.name,
        local=True,
    )
    log_object_run = mlrun_function.run(
        handler="log_object",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Run the function that will parse the data items:
    run_object = mlrun_function.run(
        handler="parse_inputs_from_type_hints",
        inputs={
            "my_data": log_dataset_run.outputs["my_list"],
            "my_encoder": log_object_run.outputs["my_object"],
            "another_data": log_dataset_run.outputs["my_array"],
        },
        params={"add": 1},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert len(run_object.outputs) == 0

    # Clean the test outputs:
    artifact_path.cleanup()


@mlrun.function(inputs={"my_data": np.ndarray})
def parse_inputs_from_wrapper(my_data, my_encoder, add, mul: int = 2):
    if isinstance(my_encoder, mlrun.DataItem):
        my_encoder = my_encoder.local()
        with open(my_encoder, "rb") as pickle_file:
            my_encoder = cloudpickle.load(pickle_file)
    return (my_encoder.transform(my_data) + add * mul).tolist()


def test_parse_inputs_from_wrapper_without_mlrun():
    """
    Run the `parse_inputs_from_wrapper` function without MLRun to see the wrapper is transparent.
    """
    _, _, _, my_data = log_dataset()
    my_encoder = log_object()
    result = parse_inputs_from_wrapper(
        pd.DataFrame(my_data), my_encoder=my_encoder, add=1
    )
    assert isinstance(result, list)
    assert result == [[2], [3], [4]]


def test_parse_inputs_from_wrapper_with_mlrun():
    """
    Run the `parse_inputs_from_wrapper` function with MLRun to see the wrapper is parsing the given inputs
    (`DataItem`s) to the written configuration provided to the wrapper.
    """
    # Create the function and run 2 of the previous functions to create a dataset and encoder objects:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    log_dataset_run = mlrun_function.run(
        handler="log_dataset",
        artifact_path=artifact_path.name,
        local=True,
    )
    log_object_run = mlrun_function.run(
        handler="log_object",
        artifact_path=artifact_path.name,
        local=True,
    )

    # Run the function that will parse the data items:
    run_object = mlrun_function.run(
        handler="parse_inputs_from_wrapper",
        inputs={
            "my_data": log_dataset_run.outputs["my_list"],
            "my_encoder": log_object_run.outputs["my_object"],
        },
        params={"add": 1},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(run_object.outputs)

    # Assertion:
    assert len(run_object.outputs) == 0  # return

    # Clean the test outputs:
    artifact_path.cleanup()
