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
import tempfile
from pathlib import Path
from typing import Union

import pytest

from mlrun.package.utils import StructFileSupportedFormat


@pytest.mark.parametrize(
    "obj",
    [
        {"a": 1, "b": 2},
        [1, 2, 3],
        [{"a": [1, 2, 3], "b": [1, 2, 3]}, {"c": [4, 5, 6]}, [1, 2, 3, 4, 5, 6]],
    ],
)
@pytest.mark.parametrize(
    "file_format",
    StructFileSupportedFormat.get_all_formats(),
)
def test_formatter(obj: Union[list, dict], file_format: str):
    """
    Test the formatters for writing and reading python objects.

    :param obj:                The object to write.
    :param file_format: The struct file format to use.
    """
    # Create a temporary directory for the test outputs:
    test_directory = tempfile.TemporaryDirectory()

    # Set up the main directory to archive and the output path for the archive file:
    file_path = Path(test_directory.name) / f"my_struct.{file_format}"
    assert not file_path.exists()

    # Archive the files:
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=obj, file_path=str(file_path))
    assert file_path.exists()

    # Extract the files:
    read_object = formatter.read(file_path=str(file_path))
    assert read_object == obj

    # Clean the test outputs:
    test_directory.cleanup()
