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
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pytest

from mlrun.package.packagers.pandas_packagers import PandasSupportedFormat


@pytest.mark.parametrize(
    "obj",
    [
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
            data=np.random.randint(0, 256, (1000, 10)),
            columns=[f"column_{i}" for i in range(10)],
        ).set_index(keys=["column_1", "column_3", "column_4"]),
    ],
)
@pytest.mark.parametrize(
    "file_format",
    PandasSupportedFormat.get_all_formats(),
)
def test_formatter(
    obj: Union[pd.DataFrame, pd.Series],
    file_format: str,
    **to_kwargs,
):
    # Create a temporary directory for the test outputs:
    test_directory = tempfile.TemporaryDirectory()

    # Set up the main directory to archive and the output path for the archive file:
    file_path = Path(test_directory.name) / f"my_array.{file_format}"
    assert not file_path.exists()

    # Save the dataframe to file:
    formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
    formatter.to(obj=obj, file_path=str(file_path), **to_kwargs)
    assert file_path.exists()

    # Read the file:
    saved_object = formatter.read(file_path=str(file_path))
    if saved_object.columns[0] == "Unnamed: 0":
        saved_object.set_index(keys=["Unnamed: 0"], drop=True, inplace=True)
        saved_object.index.set_names(names=[None], inplace=True)
    if len(obj.index.names) > 1 and len(saved_object.index.names) == 1:
        saved_object.set_index(keys=obj.index.names, inplace=True)
    assert isinstance(saved_object, type(obj))
    assert (saved_object == obj).all().all()

    # Clean the test outputs:
    test_directory.cleanup()
