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
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import mlrun


def log_artifacts_and_results() -> Tuple[
    np.ndarray, pd.DataFrame, dict, list, int, str, Pipeline
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
    return (
        np.ones((10, 20)),
        pd.DataFrame(np.zeros((20, 10))),
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        [["A"], ["B"], [""]],
        3,
        "hello",
        encoder_to_imputer,
    )


def parse_inputs(my_array, my_df, my_dict: dict, my_list, my_object, my_int, my_str):
    assert isinstance(my_array, np.ndarray)
    assert np.all(my_array == np.ones((10, 20)))

    assert isinstance(my_df, mlrun.DataItem)
    my_df = my_df.as_df()
    assert my_df.shape == (20, 10)
    assert my_df.sum().sum() == 0

    assert isinstance(my_dict, dict)
    assert my_dict == {"a": {0: 1, 1: 2, 2: 3, 3: 4}, "b": {0: 5, 1: 6, 2: 7, 3: 8}}

    assert isinstance(my_list, list)
    assert my_list == [["A"], ["B"], [""]]

    assert isinstance(my_object, Pipeline)
    assert my_object.transform(my_list).tolist() == [[0], [1], [2]]

    return [my_str] * my_int


def test_parse_inputs_from_mlrun_function():
    """
    Run the `parse_inputs_from_mlrun_function` function with MLRun to see the wrapper is parsing the given inputs
    (`DataItem`s) to the written configuration provided to the wrapper.
    """
    # Create the function and run 2 of the previous functions to create a dataset and encoder objects:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()
    log_artifacts_and_results_run = mlrun_function.run(
        handler="log_artifacts_and_results",
        returns=[
            "my_array",
            "my_df:dataset",
            {"key": "my_dict", "artifact_type": "dataset"},
            "my_list:  dataset",
            "my_int",
            "my_str : result",
            "my_object: object",
        ],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Run the function that will parse the data items:
    parse_inputs_run = mlrun_function.run(
        handler="parse_inputs",
        inputs={
            "my_list:list": log_artifacts_and_results_run.outputs["my_list"],
            "my_array : numpy.ndarray": log_artifacts_and_results_run.outputs[
                "my_array"
            ],
            "my_df": log_artifacts_and_results_run.outputs["my_df"],
            "my_object: sklearn.pipeline.Pipeline": log_artifacts_and_results_run.outputs[
                "my_object"
            ],
            "my_dict: dict": log_artifacts_and_results_run.outputs["my_dict"],
        },
        returns=["result_list: result"],
        params={
            "my_int": log_artifacts_and_results_run.outputs["my_int"],
            "my_str": log_artifacts_and_results_run.outputs["my_str"],
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Manual validation:
    mlrun.utils.logger.info(parse_inputs_run.outputs)

    # Assertion:
    assert len(parse_inputs_run.outputs) == 1  # result
    assert parse_inputs_run.outputs["result_list"] == ["hello", "hello", "hello"]

    # Clean the test outputs:
    artifact_path.cleanup()
