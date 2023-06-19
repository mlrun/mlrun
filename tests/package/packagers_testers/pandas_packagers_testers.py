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
import itertools
import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd

from mlrun.package.packagers.pandas_packagers import (
    PandasDataFramePackager,
    PandasSeriesPackager,
    PandasSupportedFormat,
)
from tests.package.packager_tester import (
    COMMON_OBJECT_INSTRUCTIONS,
    PackagerTester,
    PackTest,
    PackToUnpackTest,
    UnpackTest,
)

# Common instructions for "object" artifacts of pandas objects:
_COMMON_OBJECT_INSTRUCTIONS = {
    **COMMON_OBJECT_INSTRUCTIONS,
    "object_module_name": "pandas",
    "object_module_version": pd.__version__,
}

# Seed for reproducible tests:
np.random.seed(99)


def _prepare_result(dataframe: pd.DataFrame):
    if len(dataframe.index.names) > 1:
        orient = "split"
    elif dataframe.index.name is not None:
        orient = "dict"
    else:
        orient = "list"
    return PandasDataFramePackager._prepare_result(obj=dataframe.to_dict(orient=orient))


_DATAFRAME_SAMPLES = [
    pd.DataFrame(
        data=np.random.randint(0, 256, (1000, 10)),
        columns=[f"column_{i}" for i in range(10)],
    ),
    pd.DataFrame(
        data=np.random.randint(0, 256, (1000, 10)),
        columns=[f"column_{i}" for i in range(10)],
        index=[i for i in range(1000)],
    ),
    pd.DataFrame(
        data={
            **{f"column_{i}": np.random.randint(0, 256, 1000) for i in range(7)},
            **{f"column_{i}": np.arange(0, 1000) for i in range(7, 10)},
        },
    ).set_index(keys=["column_7", "column_8", "column_9"]),
]


def pack_dataframe(i: int) -> pd.DataFrame:
    return _DATAFRAME_SAMPLES[i]


def validate_dataframe(result: dict, i: int) -> bool:
    # Pandas dataframes are serialized as dictionaries:
    return result == _prepare_result(dataframe=_DATAFRAME_SAMPLES[i])


def unpack_dataframe(obj: pd.DataFrame, i: int):
    assert isinstance(obj, pd.DataFrame)
    assert list(obj.columns) == list(_DATAFRAME_SAMPLES[i].columns)
    assert obj.columns.names == _DATAFRAME_SAMPLES[i].columns.names
    assert obj.index.names == _DATAFRAME_SAMPLES[i].index.names
    assert (obj == _DATAFRAME_SAMPLES[i]).all().all()


def prepare_dataframe_file(file_format: str, i: int) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_dataframe.{file_format}")
    formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
    formatter.to(obj=_DATAFRAME_SAMPLES[i], file_path=file_path)
    return file_path, temp_directory


class PandasDataFramePackagerTester(PackagerTester):
    """
    A tester for the `PandasDataFramePackager`.
    """

    PACKAGER_IN_TEST = PandasDataFramePackager

    TESTS = list(
        itertools.chain.from_iterable(
            [
                *[
                    [
                        PackTest(
                            pack_handler="pack_dataframe",
                            pack_parameters={"i": i},
                            log_hint="my_result: result",
                            validation_function=validate_dataframe,
                            validation_parameters={"i": i},
                        ),
                        UnpackTest(
                            prepare_input_function=prepare_dataframe_file,
                            unpack_handler="unpack_dataframe",
                            prepare_parameters={"file_format": "parquet", "i": i},
                            unpack_parameters={"i": i},
                        ),
                        PackToUnpackTest(
                            pack_handler="pack_dataframe",
                            pack_parameters={"i": i},
                            log_hint="my_dataframe: object",
                            expected_instructions=_COMMON_OBJECT_INSTRUCTIONS,
                            unpack_handler="unpack_dataframe",
                            unpack_parameters={"i": i},
                        ),
                        PackToUnpackTest(
                            pack_handler="pack_dataframe",
                            pack_parameters={"i": i},
                            log_hint="my_dataframe: dataset",
                            unpack_handler="unpack_dataframe",
                            unpack_parameters={"i": i},
                        ),
                        *[
                            PackToUnpackTest(
                                pack_handler="pack_dataframe",
                                pack_parameters={"i": i},
                                log_hint={
                                    "key": "my_dataframe",
                                    "artifact_type": "file",
                                    "file_format": file_format,
                                },
                                expected_instructions={
                                    "file_format": file_format,
                                    "read_kwargs": {
                                        "unflatten_kwargs": {
                                            "columns": [
                                                column
                                                if not isinstance(column, tuple)
                                                else list(column)
                                                for column in _DATAFRAME_SAMPLES[
                                                    i
                                                ].columns
                                            ],
                                            "columns_levels": list(
                                                _DATAFRAME_SAMPLES[i].columns.names
                                            ),
                                            "index_levels": list(
                                                _DATAFRAME_SAMPLES[i].index.names
                                            ),
                                        }
                                    }
                                    if file_format
                                    not in [
                                        PandasSupportedFormat.PARQUET,
                                        PandasSupportedFormat.H5,
                                    ]
                                    else {},
                                },
                                unpack_handler="unpack_dataframe",
                                unpack_parameters={"i": i},
                            )
                            for file_format in ["parquet", "csv"]
                        ],
                    ]
                    for i in range(len(_DATAFRAME_SAMPLES))
                ]
            ]
        )
    )


_SERIES_SAMPLES = [
    pd.Series(data=np.random.randint(0, 256, (100,))),
    pd.Series(data=np.random.randint(0, 256, (100,)), name="my_series"),
    pd.DataFrame(data=np.random.randint(0, 256, (10, 10))).mean(),
    pd.DataFrame(data=np.random.randint(0, 256, (10, 10)))[0],
    pd.DataFrame(data=np.random.randint(0, 256, (10, 3)), columns=["a", "b", "c"])["a"],
    pd.DataFrame(
        data=np.random.randint(0, 256, (10, 4)),
        columns=["a", "b", "c", "d"],
        index=pd.MultiIndex.from_product(
            [[1, 2, 3, 4, 5], ["A", "B"]], names=["number", "letter"]
        ),
    )["a"],
]


def pack_series(i: int) -> pd.Series:
    return _SERIES_SAMPLES[i]


def validate_series(result: dict, i: int) -> bool:
    return result == _prepare_result(dataframe=pd.DataFrame(_SERIES_SAMPLES[i]))


def prepare_series_file(file_format: str, i: int) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_series.{file_format}")
    formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
    formatter.to(obj=pd.DataFrame(_SERIES_SAMPLES[i]), file_path=file_path)
    return file_path, temp_directory


def unpack_series(obj: pd.Series, i: int):
    assert isinstance(obj, pd.Series)
    assert obj.name == _SERIES_SAMPLES[i].name
    assert obj.index.names == _SERIES_SAMPLES[i].index.names
    assert (obj == _SERIES_SAMPLES[i]).all()


class PandasSeriesPackagerTester(PackagerTester):
    """
    A tester for the `PandasSeriesPackager`.
    """

    PACKAGER_IN_TEST = PandasSeriesPackager

    TESTS = list(
        itertools.chain.from_iterable(
            [
                *[
                    [
                        PackTest(
                            pack_handler="pack_series",
                            pack_parameters={"i": i},
                            log_hint="my_result: result",
                            validation_function=validate_series,
                            validation_parameters={"i": i},
                        ),
                        PackToUnpackTest(
                            pack_handler="pack_series",
                            pack_parameters={"i": i},
                            log_hint="my_dataframe: object",
                            expected_instructions=_COMMON_OBJECT_INSTRUCTIONS,
                            unpack_handler="unpack_series",
                            unpack_parameters={"i": i},
                        ),
                        PackToUnpackTest(
                            pack_handler="pack_series",
                            pack_parameters={"i": i},
                            log_hint={
                                "key": "my_series",
                                "artifact_type": "file",
                            },
                            expected_instructions={
                                "file_format": "parquet" if i in [1, 4, 5] else "csv",
                                "read_kwargs": {
                                    "unflatten_kwargs": {
                                        # Unnamed series will have a column named 0 by default when cast to dataframe.
                                        # Because we cast to dataframe before writing to file, 0 will be written for
                                        # unnamed series samples:
                                        "columns": [
                                            _SERIES_SAMPLES[i].name
                                            if _SERIES_SAMPLES[i].name is not None
                                            else 0
                                        ],
                                        "columns_levels": [None],
                                        "index_levels": list(
                                            _SERIES_SAMPLES[i].index.names
                                        ),
                                    }
                                }
                                if i not in [1, 4, 5]
                                else {},
                                "column_name": _SERIES_SAMPLES[i].name,
                            },
                            unpack_handler="unpack_series",
                            unpack_parameters={"i": i},
                        ),
                    ]
                    for i in range(len(_SERIES_SAMPLES))
                ],
                [
                    UnpackTest(
                        prepare_input_function=prepare_series_file,
                        unpack_handler="unpack_series",
                        prepare_parameters={"file_format": "parquet", "i": i},
                        unpack_parameters={"i": i},
                    )
                    for i in [1, 4, 5]
                ],
            ]
        )
    )
