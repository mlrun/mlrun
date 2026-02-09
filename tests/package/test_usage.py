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
import copy
import json
import os
import pathlib
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import mlrun
from mlrun import DataItem
from mlrun.package import DefaultPackager
from tests.package.usage_assets import BaseClass, InheritingClass

RETURNS_LOG_HINTS = [
    "my_array",
    "my_df",
    {"key": "my_dict", "artifact_type": "object"},
    "my_list:  file",
    "my_int",
    "my_str : result",
    "my_object: object",
    "*my_df_dict",
    "*my_array_list",
]


def log_artifacts_and_results() -> (
    tuple[
        np.ndarray,
        pd.DataFrame,
        dict,
        list,
        int,
        str,
        Pipeline,
        dict[str, pd.DataFrame],
        list[np.ndarray],
    ]
):
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
    context.log_artifact(item="manually_logged_file", local_path=file_path)

    dataframes = {
        "abc": pd.DataFrame(np.random.rand(5, 5)),
        "def": pd.DataFrame(np.random.rand(10, 3)),
        "ghi": pd.DataFrame(np.random.rand(7, 8)),
    }
    arrays = [np.random.rand(10) for _ in range(10)]

    return (
        np.ones((10, 20)),
        pd.DataFrame(np.zeros((20, 10))),
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        [["A"], ["B"], [""]],
        3,
        "hello",
        encoder_to_imputer,
        dataframes,
        arrays,
    )


def _assert_parsing(
    my_array: np.ndarray,
    my_df: mlrun.DataItem,
    manually_logged_file: str | mlrun.DataItem,
    my_dict: dict,
    my_list: list,
    my_object: Pipeline,
    my_int: int,
    my_str: str,
    my_df_dict: dict[str, pd.DataFrame],
    my_array_list: list[np.ndarray],
):
    assert isinstance(my_array, np.ndarray)
    assert np.all(my_array == np.ones((10, 20)))

    assert isinstance(my_df, mlrun.DataItem)
    my_df = my_df.as_df()
    assert my_df.shape == (20, 10)
    assert my_df.sum().sum() == 0

    assert isinstance(manually_logged_file, mlrun.DataItem)
    manually_logged_file = manually_logged_file.local()
    with open(manually_logged_file) as file:
        file_content = file.read()
    assert file_content == "123"

    assert isinstance(my_dict, dict)
    assert my_dict == {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}

    assert isinstance(my_list, list)
    assert my_list == [["A"], ["B"], [""]]

    assert isinstance(my_object, Pipeline)
    assert my_object.transform(my_list).tolist() == [[0], [1], [2]]

    assert isinstance(my_df_dict, dict)
    assert list(my_df_dict.keys()) == ["abc", "def", "ghi"]
    for v in my_df_dict.values():
        assert isinstance(v, pd.DataFrame)

    assert isinstance(my_array_list, list)
    for v in my_array_list:
        assert isinstance(
            v,
            np.ndarray if mlrun.mlconf.packagers.auto_unpack_inputs else mlrun.DataItem,
        )

    return [my_str] * my_int


def parse_inputs_from_type_annotations(
    my_array: np.ndarray,
    my_df: mlrun.DataItem,
    manually_logged_file: str | mlrun.DataItem,
    my_dict: dict,
    my_list: list,
    my_object: Pipeline,
    my_int: int,
    my_str: str,
    my_df_dict: dict[str, pd.DataFrame],
    my_array_list: list,
):
    _assert_parsing(
        my_array=my_array,
        my_df=my_df,
        manually_logged_file=manually_logged_file,
        my_dict=my_dict,
        my_list=my_list,
        my_object=my_object,
        my_int=my_int,
        my_str=my_str,
        my_df_dict=my_df_dict,
        my_array_list=my_array_list,
    )


def parse_inputs_from_mlrun_function(
    my_array,
    my_df,
    manually_logged_file,
    my_dict,
    my_list,
    my_object,
    my_int,
    my_str,
    my_df_dict,
    my_array_list,
    is_conf_test: bool,
):
    if is_conf_test:
        _assert_auto_unpacking(
            my_array=my_array,
            my_df=my_df,
            manually_logged_file=manually_logged_file,
            my_dict=my_dict,
            my_list=my_list,
            my_object=my_object,
            my_df_dict=my_df_dict,
            my_array_list=my_array_list,
        )
    else:
        _assert_parsing(
            my_array=my_array,
            my_df=my_df,
            manually_logged_file=manually_logged_file,
            my_dict=my_dict,
            my_list=my_list,
            my_object=my_object,
            my_int=my_int,
            my_str=my_str,
            my_df_dict=my_df_dict,
            my_array_list=my_array_list,
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

    # There should always be at least two outputs - the manually logged result and artifact:
    if is_enabled and returns:
        # Plus all configured returning values ("*my_df_dict_" yields 4 outputs + "*my_array_list_" yields 11
        # outputs - 2, the keys):
        assert (
            len(log_artifacts_and_results_run.outputs)
            == 2 + len(RETURNS_LOG_HINTS) + 4 + 11 - 2
        )
    else:
        # Plus the default logged output as string MLRun did before packagers and log hints:
        assert len(log_artifacts_and_results_run.outputs) == 2 + 1


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
            "manually_logged_file": log_artifacts_and_results_run.outputs[
                "manually_logged_file"
            ],
            "my_object": log_artifacts_and_results_run.outputs["my_object"],
            "my_dict": log_artifacts_and_results_run.outputs["my_dict"],
            "my_df_dict": {
                "abc": log_artifacts_and_results_run.outputs["my_df_dict_abc"],
                "def": log_artifacts_and_results_run.outputs["my_df_dict_def"],
                "ghi": log_artifacts_and_results_run.outputs["my_df_dict_ghi"],
            },
            "my_array_list": [
                log_artifacts_and_results_run.outputs["my_array_list_0"],
                log_artifacts_and_results_run.outputs["my_array_list_1"],
                log_artifacts_and_results_run.outputs["my_array_list_2"],
                log_artifacts_and_results_run.outputs["my_array_list_3"],
                log_artifacts_and_results_run.outputs["my_array_list_4"],
                log_artifacts_and_results_run.outputs["my_array_list_5"],
                log_artifacts_and_results_run.outputs["my_array_list_6"],
                log_artifacts_and_results_run.outputs["my_array_list_7"],
                log_artifacts_and_results_run.outputs["my_array_list_8"],
                log_artifacts_and_results_run.outputs["my_array_list_9"],
            ],
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
            "my_df : mlrun.DataItem": log_artifacts_and_results_run.outputs["my_df"],
            "manually_logged_file": log_artifacts_and_results_run.outputs[
                "manually_logged_file"
            ],
            "my_object: sklearn.pipeline.Pipeline": log_artifacts_and_results_run.outputs[
                "my_object"
            ],
            "my_dict: dict": log_artifacts_and_results_run.outputs["my_dict"],
            "my_df_dict: dict[pandas.DataFrame]": {
                "abc": log_artifacts_and_results_run.outputs["my_df_dict_abc"],
                "def": log_artifacts_and_results_run.outputs["my_df_dict_def"],
                "ghi": log_artifacts_and_results_run.outputs["my_df_dict_ghi"],
            },
            "my_array_list: list": [
                log_artifacts_and_results_run.outputs["my_array_list_0"],
                log_artifacts_and_results_run.outputs["my_array_list_1"],
                log_artifacts_and_results_run.outputs["my_array_list_2"],
                log_artifacts_and_results_run.outputs["my_array_list_3"],
                log_artifacts_and_results_run.outputs["my_array_list_4"],
                log_artifacts_and_results_run.outputs["my_array_list_5"],
                log_artifacts_and_results_run.outputs["my_array_list_6"],
                log_artifacts_and_results_run.outputs["my_array_list_7"],
                log_artifacts_and_results_run.outputs["my_array_list_8"],
                log_artifacts_and_results_run.outputs["my_array_list_9"],
            ],
        },
        params={
            "my_int": log_artifacts_and_results_run.outputs["my_int"],
            "my_str": log_artifacts_and_results_run.outputs["my_str"],
            "is_conf_test": False,
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Clean the test outputs:
    artifact_path.cleanup()


def _assert_auto_unpacking(
    my_array,
    my_df,
    manually_logged_file,
    my_dict,
    my_list,
    my_object,
    my_df_dict,
    my_array_list,
):
    if not mlrun.mlconf.packagers.auto_unpack_inputs:
        # Make sure all inputs are DataItems (were not unpacked):
        for obj in [my_array, my_df, manually_logged_file, my_dict, my_list, my_object]:
            assert isinstance(obj, mlrun.DataItem)

        for v in my_df_dict.values():
            assert isinstance(v, mlrun.DataItem)

        for v in my_array_list:
            assert isinstance(v, mlrun.DataItem)
    else:
        assert isinstance(my_array, np.ndarray)
        assert isinstance(my_df, pd.DataFrame)
        assert isinstance(
            manually_logged_file, mlrun.DataItem
        )  # Not logged via packager.
        assert isinstance(my_dict, dict)
        assert isinstance(my_list, list)
        assert isinstance(my_object, Pipeline)
        for v in my_df_dict.values():
            assert isinstance(v, pd.DataFrame)
        for v in my_array_list:
            assert isinstance(v, np.ndarray)


@pytest.mark.parametrize("auto_unpack_inputs", [True, False])
def test_parse_inputs_with_mlconf_packagers_auto_unpack_inputs(
    rundb_mock, auto_unpack_inputs: bool
):
    """
    Run the `parse_inputs_from_mlrun_function` function with MLRun to see the packagers are being auto-unpacked by their
    original packager or not if auto_unpack_inputs is not set - which means all will remain `DataItem`s.

    :param rundb_mock:         A runDB mock fixture.
    :param auto_unpack_inputs: The `mlconf.packagers.auto_unpack_inputs` configuration value.
    """
    # Set the configuration:
    mlrun.mlconf.packagers.auto_unpack_inputs = auto_unpack_inputs

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
            "my_list": log_artifacts_and_results_run.outputs["my_list"],
            "my_array": log_artifacts_and_results_run.outputs["my_array"],
            "my_df": log_artifacts_and_results_run.outputs["my_df"],
            "manually_logged_file": log_artifacts_and_results_run.outputs[
                "manually_logged_file"
            ],
            "my_object": log_artifacts_and_results_run.outputs["my_object"],
            "my_dict": log_artifacts_and_results_run.outputs["my_dict"],
            "my_df_dict": log_artifacts_and_results_run.outputs["my_df_dict"],
            "my_array_list": log_artifacts_and_results_run.outputs["my_array_list"],
        },
        params={
            "my_int": log_artifacts_and_results_run.outputs["my_int"],
            "my_str": log_artifacts_and_results_run.outputs["my_str"],
            "is_conf_test": True,
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Clean the test outputs:
    artifact_path.cleanup()


def log_with_and_without_packagers():
    context = mlrun.get_or_create_ctx(name="ctx")

    context_df = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
    package_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    context.log_dataset("context_df", context_df)

    return package_df


def parse_from_package_and_context(context_df, package_df):
    if mlrun.mlconf.packagers.auto_unpack_inputs:
        assert isinstance(context_df, mlrun.DataItem)
        assert context_df.as_df().equals(
            pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        )
        assert isinstance(package_df, pd.DataFrame)
        assert package_df.equals(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    else:
        assert isinstance(context_df, mlrun.DataItem)
        assert context_df.as_df().equals(
            pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        )
        assert isinstance(package_df, mlrun.DataItem)
        assert package_df.as_df().equals(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))


@pytest.mark.parametrize("auto_unpack_inputs", [True, False])
def test_log_with_and_without_packagers(rundb_mock, auto_unpack_inputs: bool):
    """
    Run the `log_with_and_without_packagers` function with MLRun to see the packagers are being auto-unpacked by their
    original packager or not if auto_unpack_inputs is not set or the input is not a package (logged via context).

    :param rundb_mock:         A runDB mock fixture.
    :param auto_unpack_inputs: The `mlconf.packagers.auto_unpack_inputs` configuration value.
    """
    # Set the configuration:
    mlrun.mlconf.packagers.auto_unpack_inputs = auto_unpack_inputs

    # Create the function:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()

    # Run the logging functions:
    log_with_and_without_packagers_run = mlrun_function.run(
        handler="log_with_and_without_packagers",
        returns=["package_df"],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Run the function that will parse the data items:
    mlrun_function.run(
        handler="parse_from_package_and_context",
        inputs={
            "context_df": log_with_and_without_packagers_run.outputs["context_df"],
            "package_df": log_with_and_without_packagers_run.outputs["package_df"],
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Clean the test outputs:
    artifact_path.cleanup()


@pytest.mark.parametrize("auto_pack_outputs", [True, False])
@pytest.mark.parametrize("auto_pack_key", [None, "test"])
@pytest.mark.parametrize(
    "returns",
    [
        [
            "my_array",
            "my_df",
            {"key": "my_dict", "artifact_type": "object"},
        ],
        None,
    ],
)
def test_log_outputs_with_mlconf_packagers_auto_pack_outputs(
    rundb_mock,
    auto_pack_outputs: bool,
    auto_pack_key: str,
    returns: list[str],
):
    """
    Run the `log_artifacts_and_results` function with MLRun to see the packagers are packing the outputs automatically
    given the `mlconf.packagers.auto_pack_outputs` configuration in different scenarios.

    :param rundb_mock:        A runDB mock fixture.
    :param auto_pack_outputs: The `mlconf.packagers.auto_pack_outputs` configuration value.
    :param auto_pack_key:     The `mlconf.packagers.auto_pack_key` configuration value.
    :param returns:           Log hints to pass in the 'returns' parameter.
    """
    # Set the configuration:
    mlrun.mlconf.packagers.auto_pack_outputs = auto_pack_outputs
    if auto_pack_key:
        mlrun.mlconf.packagers.auto_pack_key = auto_pack_key

    # Create the function:
    mlrun_function = mlrun.code_to_function(filename=__file__, kind="job")
    artifact_path = tempfile.TemporaryDirectory()

    # Run the logging function:
    log_artifacts_and_results_run = mlrun_function.run(
        name="test-run",
        handler="log_artifacts_and_results",
        artifact_path=artifact_path.name,
        returns=copy.deepcopy(
            returns
        ),  # We copy as in local the outputs are being appended to the list.
        local=True,
    )

    # There should always be at least two outputs - the manually logged result and artifact:
    if auto_pack_outputs:
        # 'auto_pack_outputs' is set, assert all returning values (notice there are no '*' in "my_df_dict_" and
        # "my_array_list_" now):
        assert len(log_artifacts_and_results_run.outputs) == 2 + len(RETURNS_LOG_HINTS)
        # Check for the manually logged result and artifact:
        assert "manually_logged_result" in log_artifacts_and_results_run.outputs
        assert "manually_logged_file" in log_artifacts_and_results_run.outputs
        # Check the auto-logged outputs:
        key = "test-run-test" if auto_pack_key else "test-run-artifact"
        for i in range(
            len(log_artifacts_and_results_run.outputs)
            - (2 + (len(returns) if returns else 0))
        ):
            assert f"{key}-{i}" in log_artifacts_and_results_run.outputs
        # Check the outputs from the 'returns' parameter:
        if returns:
            for log_hint in returns:
                if isinstance(log_hint, str):
                    assert (
                        log_hint.split(":")[0].strip()
                        in log_artifacts_and_results_run.outputs
                    )
                elif isinstance(log_hint, dict):
                    assert log_hint["key"] in log_artifacts_and_results_run.outputs
    else:
        # Plus the default logged output as string MLRun did before packagers and log hints if returns is not set:
        assert len(log_artifacts_and_results_run.outputs) == 2 + (
            len(returns) if returns else 1
        )


class BaseClassPackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = BaseClass
    PACK_SUBCLASSES = True

    def unpack_object(
        self,
        data_item: DataItem,
        pickle_module_name: str = "cloudpickle",
        object_module_name: str | None = None,
        python_version: str | None = None,
        pickle_module_version: str | None = None,
        object_module_version: str | None = None,
    ) -> Any:
        base_class = super().unpack_object(
            data_item=data_item,
            pickle_module_name=pickle_module_name,
            object_module_name=object_module_name,
            python_version=python_version,
            pickle_module_version=pickle_module_version,
            object_module_version=object_module_version,
        )

        # To make sure this packager unpacked the object and not the default packager, we reduce a by 1 (it will be
        # asserted in the unpacking functions):
        base_class.a -= 1

        return base_class


def func_to_pack_base_class(a: int, b: str | None = None) -> BaseClass:
    if b:
        return InheritingClass(a=a, b=b)
    return BaseClass(a=a)


def func_to_unpack_base_class(base_class: BaseClass, a: int, b: str | None = None):
    assert isinstance(base_class, BaseClass)
    assert base_class.a == a - 1


def func_to_unpack_inheriting_class(
    base_class: InheritingClass, a: int, b: str | None = None
):
    assert isinstance(base_class, InheritingClass)
    assert base_class.b == b
    assert base_class.a == a - 1


@pytest.mark.parametrize("a, b", [(10, None), (1, "a test")])
def test_subclasses_packing_and_unpacking(rundb_mock, a: int, b: str):
    """
    Run the `func_to_pack_base_class` and `func_to_unpack_base_class` functions with MLRun to see the custom `BaseClass`
    packager is packing and unpacking subclasses successfully.

    :param rundb_mock: A runDB mock fixture.
    :param a:          The `a` value for the `BaseClass` init method.
    :param b:          The `b` value for the `InheritingClass` init method. None will mark the test to initialize a
                       `BaseClass`.
    """
    # Get the project:
    project = mlrun.get_or_create_project("test-project", allow_cross_project=True)

    # Add the custom packager for `BaseClass`:
    project.add_custom_packager(
        packager="tests.package.test_usage.BaseClassPackager", is_mandatory=True
    )
    project.save()

    # Create the function:
    mlrun_function = project.set_function(
        func=__file__, name="test_func", kind="job", image="mlrun/mlrun"
    )
    artifact_path = tempfile.TemporaryDirectory()

    # Run the packing function:
    pack_run = mlrun_function.run(
        handler="func_to_pack_base_class",
        params={"a": a, "b": b},
        returns=["base_class"],
        artifact_path=artifact_path.name,
        local=True,
    )

    # Make sure the `BaseClassPackager` packed the object:
    unpacking_instructions = pack_run.status.artifacts[0]["spec"][
        "unpackaging_instructions"
    ]
    assert unpacking_instructions["packager_name"] == BaseClassPackager.__name__
    assert unpacking_instructions["object_type"] == (
        f"{InheritingClass.__module__}.{InheritingClass.__name__}"
        if b
        else f"{BaseClass.__module__}.{BaseClass.__name__}"
    )

    # Run the unpacking function:
    mlrun_function.run(
        handler="func_to_unpack_inheriting_class" if b else "func_to_unpack_base_class",
        inputs={"base_class": pack_run.outputs["base_class"]},
        params={
            "a": a,
            "b": b,
        },
        artifact_path=artifact_path.name,
        local=True,
    )

    # Clean the test outputs:
    artifact_path.cleanup()


_JSON_SAMPLE = {"a": 1, "b": 2}


def parse_local_file(my_dict: dict):
    assert isinstance(my_dict, dict)
    assert my_dict == _JSON_SAMPLE


def test_parse_local_file(rundb_mock):
    """
    Run the `parse_local_file` function with MLRun to verify the json file given for it to parse as dictionary will not
    be deleted as it is a local path.

    :param rundb_mock: A runDB mock fixture.
    """
    # Get the project:
    project = mlrun.get_or_create_project("default", allow_cross_project=True)

    # Create a json file of a dictionary:
    artifact_path = tempfile.TemporaryDirectory()
    json_path = pathlib.Path(artifact_path.name) / "my_dict.json"
    with open(json_path, "w") as file:
        json.dump(_JSON_SAMPLE, file)
    assert json_path.exists()

    # Create the function:
    mlrun_function = project.set_function(
        func=__file__, name="test_func", kind="job", image="mlrun/mlrun"
    )

    # Run the packing function:
    mlrun_function.run(
        handler="parse_local_file",
        inputs={"my_dict": str(json_path)},
        artifact_path=artifact_path.name,
        local=True,
    )

    # Make sure the file was not deleted post run:
    assert json_path.exists()

    # Make sure the file was not changed
    with open(json_path) as file:
        my_dict = json.load(file)
    assert my_dict == _JSON_SAMPLE

    # Clean the test outputs:
    artifact_path.cleanup()
