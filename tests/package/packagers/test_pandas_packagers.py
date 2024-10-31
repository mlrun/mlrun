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
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrun.package.packagers.pandas_packagers import PandasSupportedFormat

# Set up the format requirements dictionary:
FORMAT_REQUIREMENTS = {
    PandasSupportedFormat.PARQUET: "pyarrow",
    PandasSupportedFormat.H5: "tables",
    PandasSupportedFormat.XLSX: "openpyxl",
    PandasSupportedFormat.XML: "lxml",
    PandasSupportedFormat.HTML: "lxml",
    PandasSupportedFormat.FEATHER: "pyarrow",
    PandasSupportedFormat.ORC: "pyarrow",
}


def check_skipping_pandas_format(fmt: str):
    if fmt in FORMAT_REQUIREMENTS:
        try:
            importlib.import_module(FORMAT_REQUIREMENTS[fmt])
        except ModuleNotFoundError:
            return True

    # TODO: Remove when padnas>=1.5 in requirements
    if fmt == PandasSupportedFormat.ORC:
        # ORC format is added only since pandas 1.5.0, so we skip if pandas is older than this:
        v1, v2, v3 = pd.__version__.split(".")
        if int(v1) == 1 and int(v2) < 5:
            return True
    return False


def get_test_dataframes():
    # Configurations:
    _n_rows = 100
    _n_columns = 24
    _single_level_column_names = [f"column_{i}" for i in range(_n_columns)]
    _multi_level_column_names = [
        [f"{chr(n)}1" for n in range(ord("A"), ord("A") + 2)],
        [f"{chr(n)}2" for n in range(ord("A"), ord("A") + 3)],
        [f"{chr(n)}3" for n in range(ord("A"), ord("A") + 4)],
    ]  # 2 * 3 * 4 = 24 (_n_columns)
    _column_levels_names = ["letter_level_1", "letter_level_2", "letter_level_3"]
    _single_index = [i for i in range(0, _n_rows * 2, 2)]
    _multi_index = [
        list(range(2)),
        list(range(5)),
        list(range(10)),
    ]  # 2 * 5 * 10 = 100 (_n_rows)

    # Initialize the data and options for dataframes:
    data = np.random.randint(0, 256, (_n_rows, _n_columns))
    columns_options = [
        # Single level:
        _single_level_column_names,
        # Multi-level:
        pd.MultiIndex.from_product(_multi_level_column_names),
        # Multi-level with names:
        pd.MultiIndex.from_product(
            _multi_level_column_names,
            names=_column_levels_names,
        ),
    ]
    index_options = [
        # Default:
        None,
        # Single level:
        _single_index,
        # Single level with name:
        pd.Index(data=_single_index, name="my_index"),
        # Multi-level:
        pd.MultiIndex.from_product(_multi_index),
        # Multi-level with names:
        pd.MultiIndex.from_product(
            _multi_index, names=["index_5", "index_10", "index_20"]
        ),
    ]

    # Initialize the dataframes:
    dataframes = []
    for columns in columns_options:
        for index in index_options:
            df = pd.DataFrame(data=data, columns=columns, index=index)
            dataframes.append(df)
            # Add same name of columns and indexes scenarios if index has a name:
            if index is not None and all(
                index_name is not None for index_name in df.index.names
            ):
                same_name_df = df.copy()
                if isinstance(df.index, pd.MultiIndex):
                    if isinstance(df.columns, pd.MultiIndex):
                        same_name_df.index.set_names(
                            names=df.columns.names[: len(df.index.names)], inplace=True
                        )
                    else:  # Single index
                        same_name_df.index.set_names(
                            names=df.columns[: len(df.index.names)], inplace=True
                        )
                else:  # Single index
                    if isinstance(df.columns, pd.MultiIndex):
                        same_name_df.index.set_names(
                            names=str(df.columns.names[0]), inplace=True
                        )
                    else:  # Single index
                        same_name_df.index.set_names(
                            names=str(df.columns[0]), inplace=True
                        )
                dataframes.append(same_name_df)

    return dataframes


@pytest.mark.parametrize("obj", get_test_dataframes())
@pytest.mark.parametrize(
    "file_format",
    PandasSupportedFormat.get_all_formats(),
)
def test_formatter(
    obj: pd.DataFrame,
    file_format: str,
):
    """
    Test the pandas formatters for writing and reading dataframes.

    :param obj:         The dataframe to write.
    :param file_format: The pandas file format to use.
    """
    # Check if needed to skip this file format test:
    if check_skipping_pandas_format(fmt=file_format):
        pytest.skip(
            f"Skipping test of pandas file format '{file_format}' "
            f"due to missing requirement: '{FORMAT_REQUIREMENTS[file_format]}'"
        )

    # Create a temporary directory for the test outputs:
    test_directory = tempfile.TemporaryDirectory()

    # Set up the main directory to archive and the output path for the archive file:
    file_path = Path(test_directory.name) / f"my_array.{file_format}"
    assert not file_path.exists()

    # Save the dataframe to file:
    formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
    read_kwargs = formatter.to(obj=obj.copy(), file_path=str(file_path))
    assert file_path.exists()

    # Read the file:
    saved_object = formatter.read(file_path=str(file_path), **read_kwargs)

    # Assert equality post reading:
    assert isinstance(saved_object, type(obj))
    assert list(saved_object.columns) == list(obj.columns)
    assert saved_object.columns.names == obj.columns.names
    assert saved_object.index.names == obj.index.names
    assert (saved_object == obj).all().all()

    # Clean the test outputs:
    test_directory.cleanup()
