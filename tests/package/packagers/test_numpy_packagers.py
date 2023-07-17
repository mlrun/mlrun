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
from typing import Dict, List, Union

import numpy as np
import pytest

from mlrun.package.packagers.numpy_packagers import NumPySupportedFormat


def _test(
    obj: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
    file_format: str,
    **save_kwargs,
):
    # Create a temporary directory for the test outputs:
    test_directory = tempfile.TemporaryDirectory()

    # Set up the main directory to archive and the output path for the archive file:
    file_path = Path(test_directory.name) / f"my_array.{file_format}"
    assert not file_path.exists()

    # Archive the files:
    formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
    formatter.save(obj=obj, file_path=str(file_path), **save_kwargs)
    assert file_path.exists()

    # Extract the files:
    saved_object = formatter.load(file_path=str(file_path))
    if isinstance(obj, np.ndarray):
        assert (saved_object == obj).all()
    elif isinstance(obj, dict):
        for original, saved in zip(obj.values(), saved_object.values()):
            assert (original == saved).all()
    else:
        for original, saved in zip(obj, saved_object.values()):
            assert (original == saved).all()

    # Clean the test outputs:
    test_directory.cleanup()


@pytest.mark.parametrize(
    "obj",
    [
        np.random.random((10, 30)),
        np.random.random(100),
        np.random.randint(0, 255, (150, 200)),
    ],
)
@pytest.mark.parametrize(
    "file_format",
    NumPySupportedFormat.get_single_array_formats(),
)
def test_formatter_single_array(obj: np.ndarray, file_format: str):
    """
    Test the formatters for saving and writing a numpy array.

    :param obj:         The array to write.
    :param file_format: The numpy format to use.
    """
    _test(file_format=file_format, obj=obj)


@pytest.mark.parametrize(
    "obj",
    [
        {f"array_{i}": np.random.random(size=(10, 30)) for i in range(5)},
        [np.random.random(size=777) for i in range(10)],
    ],
)
@pytest.mark.parametrize(
    "file_format",
    NumPySupportedFormat.get_multi_array_formats(),
)
@pytest.mark.parametrize(
    "save_kwargs", [{"is_compressed": boolean_value} for boolean_value in [True, False]]
)
def test_formatter_multiple_arrays(
    obj: Union[Dict[str, np.ndarray], List[np.ndarray]],
    file_format: str,
    save_kwargs: bool,
):
    """
    Test the formatters for saving and writing a numpy array.

    :param obj:         The array to write.
    :param file_format: The numpy format to use.
    :param save_kwargs: Save kwargs to use.
    """
    _test(obj=obj, file_format=file_format, save_kwargs=save_kwargs)
