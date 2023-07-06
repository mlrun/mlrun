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
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import mlrun

RETURNS_LOG_HINTS = [
    "my_array",
    "my_df",
    "my_file: path",
    {"key": "my_dict", "artifact_type": "object"},
    "my_list:  file",
    "my_int",
    "my_str : result",
    "my_object: object",
]


def log_artifacts_and_results() -> Tuple[
    np.ndarray, pd.DataFrame, str, dict, list, int, str, Pipeline
]:
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

    context = mlrun.get_or_create_ctx(name="ctx")
    context.log_result(key="manually_logged_result", value=10)

    file_path = os.path.join(context.artifact_path, "my_file.txt")
    with open(file_path, "w") as file:
        file.write("123")

    return (
        np.ones((10, 20)),
        pd.DataFrame(np.zeros((20, 10))),
        file_path,
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        [["A"], ["B"], [""]],
        3,
        "hello",
        encoder_to_imputer,
    )


def _assert_parsing(
    my_array: np.ndarray,
    my_df: mlrun.DataItem,
    my_file: Union[int, mlrun.DataItem],
    my_dict: dict,
    my_list: list,
    my_object: Pipeline,
    my_int: int,
    my_str: str,
):
    assert isinstance(my_array, np.ndarray)
    assert np.all(my_array == np.ones((10, 20)))

    assert isinstance(my_df, mlrun.DataItem)
    my_df = my_df.as_df()
    assert my_df.shape == (20, 10)
    assert my_df.sum().sum() == 0

    assert isinstance(my_file, mlrun.DataItem)
    my_file = my_file.local()
    with open(my_file, "r") as file:
        file_content = file.read()
    assert file_content == "123"

    assert isinstance(my_dict, dict)
    assert my_dict == {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}

    assert isinstance(my_list, list)
    assert my_list == [["A"], ["B"], [""]]

    assert isinstance(my_object, Pipeline)
    assert my_object.transform(my_list).tolist() == [[0], [1], [2]]

    return [my_str] * my_int


def parse_inputs_from_type_annotations(
    my_array: np.ndarray,
    my_df: mlrun.DataItem,
    my_file: Union[int, mlrun.DataItem],
    my_dict: dict,
    my_list: list,
    my_object: Pipeline,
    my_int: int,
    my_str: str,
):
    _assert_parsing(
        my_array=my_array,
        my_df=my_df,
        my_file=my_file,
        my_dict=my_dict,
        my_list=my_list,
        my_object=my_object,
        my_int=my_int,
        my_str=my_str,
    )


def parse_inputs_from_mlrun_function(
    my_array, my_df, my_file, my_dict, my_list, my_object, my_int, my_str
):
    _assert_parsing(
        my_array=my_array,
        my_df=my_df,
        my_file=my_file,
        my_dict=my_dict,
        my_list=my_list,
        my_object=my_object,
        my_int=my_int,
        my_str=my_str,
    )


@pytest.mark.parametrize("is_enabled", [True, False])
@pytest.mark.parametrize("returns", [RETURNS_LOG_HINTS, []])
def test_mlconf_packagers_enabled(rundb_mock, is_enabled: bool, returns: list):
    """
    Test the packagers logging given the returns parameter in the `run` method and MLRun's `mlconf.packagers.enabled`
    configuration.

    :param rundb_mock: A runDB mock fixture.
    :param is_enabled: The `mlconf.packagers.enabled` configuration value.
    :param returns:    Log hints to pass in the 'returns' parameter.
    """
    # Set the configuration:
    mlrun.mlconf.packagers.enabled = is_enabled

    # Create the function:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()

    # Run the logging function:
    log_artifacts_and_results_run = mlrun_function.run(
        handler="log_artifacts_and_results",
        returns=returns,
        artifact_path=artifact_path.name,
        local=True,
    )

    # There should always be at least one output - the manually logged result:
    if is_enabled and returns:
        # Plus all configured returning values:
        assert len(log_artifacts_and_results_run.outputs) == 1 + len(RETURNS_LOG_HINTS)
    else:
        # Plus the default logged output as string MLRun did before packagers and log hints:
        assert len(log_artifacts_and_results_run.outputs) == 1 + 1


def test_parse_inputs_from_type_annotations(rundb_mock):
    """
    Run the `parse_inputs_from_type_annotations` function with MLRun to see the packagers are parsing the given inputs
    (`DataItem`s) to the written type hints.

    :param rundb_mock: A runDB mock fixture.
    """
    # Create the function:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()

    # Run the logging functions:
    log_artifacts_and_results_run = mlrun_function.run(
        handler="log_artifacts_and_results",
        returns=RETURNS_LOG_HINTS,
        artifact_path=artifact_path.name,
        local=True,
    )

    # Run the function that will parse the data items:
    mlrun_function.run(
        handler="parse_inputs_from_type_annotations",
        inputs={
            "my_list": log_artifacts_and_results_run.outputs["my_list"],
            "my_array": log_artifacts_and_results_run.outputs["my_array"],
            "my_df": log_artifacts_and_results_run.outputs["my_df"],
            "my_file": log_artifacts_and_results_run.outputs["my_file"],
            "my_object": log_artifacts_and_results_run.outputs["my_object"],
            "my_dict": log_artifacts_and_results_run.outputs["my_dict"],
        },
        params={
            "my_int": log_artifacts_and_results_run.outputs["my_int"],
            "my_str": log_artifacts_and_results_run.outputs["my_str"],
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Clean the test outputs:
    artifact_path.cleanup()


def test_parse_inputs_from_mlrun_function(rundb_mock):
    """
    Run the `parse_inputs_from_mlrun_function` function with MLRun to see the packagers are parsing the given inputs
    (`DataItem`s) to the provided configuration in the `run` method.

    :param rundb_mock: A runDB mock fixture.
    """
    # Create the function:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()

    # Run the logging functions:
    log_artifacts_and_results_run = mlrun_function.run(
        handler="log_artifacts_and_results",
        returns=RETURNS_LOG_HINTS,
        artifact_path=artifact_path.name,
        local=True,
    )

    # Run the function that will parse the data items:
    mlrun_function.run(
        handler="parse_inputs_from_mlrun_function",
        inputs={
            "my_list:list": log_artifacts_and_results_run.outputs["my_list"],
            "my_array : numpy.ndarray": log_artifacts_and_results_run.outputs[
                "my_array"
            ],
            "my_df": log_artifacts_and_results_run.outputs["my_df"],
            "my_file": log_artifacts_and_results_run.outputs["my_file"],
            "my_object: sklearn.pipeline.Pipeline": log_artifacts_and_results_run.outputs[
                "my_object"
            ],
            "my_dict: dict": log_artifacts_and_results_run.outputs["my_dict"],
        },
        params={
            "my_int": log_artifacts_and_results_run.outputs["my_int"],
            "my_str": log_artifacts_and_results_run.outputs["my_str"],
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Clean the test outputs:
    artifact_path.cleanup()
