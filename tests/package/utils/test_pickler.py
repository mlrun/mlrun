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

import cloudpickle
import numpy as np
import pytest

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package.utils import Pickler


@pytest.mark.parametrize(
    "pickle_module_name, expected_notes",
    [
        (
            "pickle",
            {
                "object_module_name": "numpy",
                "pickle_module_name": "pickle",
                "python_version": Pickler._get_python_version(),
                "object_module_version": np.__version__,
            },
        ),
        (
            "cloudpickle",
            {
                "object_module_name": "numpy",
                "pickle_module_name": "cloudpickle",
                "python_version": Pickler._get_python_version(),
                "object_module_version": np.__version__,
                "pickle_module_version": cloudpickle.__version__,
            },
        ),
        ("numpy", "A pickle module is expected to have a"),
    ],
)
def test_pickler(pickle_module_name: str, expected_notes: Union[dict, str]):
    """
    Test the `Pickler` with multiple pickling modules.

    :param pickle_module_name: The pickle module name to use.
    :param expected_notes:     The expected pickling notes. A string value indicates the `Pickler` should fail with the
                               provided error message in the variable.
    """
    # Create the test temporary directory:
    test_directory = tempfile.TemporaryDirectory()

    # Prepare the pickle path and the object to pickle:
    output_path = Path(test_directory.name) / "my_array.pkl"
    array = np.random.random(size=100)

    # Pickle:
    try:
        _, notes = Pickler.pickle(
            obj=array,
            pickle_module_name=pickle_module_name,
            output_path=str(output_path),
        )
    except MLRunInvalidArgumentError as error:
        if isinstance(expected_notes, str):
            assert expected_notes in str(error)
            return
        raise error
    assert output_path.exists()
    assert notes == expected_notes

    # Unpickle:
    pickled_array = Pickler.unpickle(pickle_path=str(output_path), **notes)
    np.testing.assert_equal(pickled_array, array)

    # Delete the test directory (with the pickle file that was created):
    test_directory.cleanup()
