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
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np

from mlrun.package.packagers.numpy_packagers import (
    NumPyNDArrayDictPackager,
    NumPyNDArrayListPackager,
    NumPyNDArrayPackager,
    NumPyNumberPackager,
    NumPySupportedFormat,
)
from tests.package.packager_tester import (
    COMMON_OBJECT_INSTRUCTIONS,
    PackagerTester,
    PackTest,
    PackToUnpackTest,
    UnpackTest,
)

# Common instructions for "object" artifacts of numpy objects:
_COMMON_OBJECT_INSTRUCTIONS = {
    **COMMON_OBJECT_INSTRUCTIONS,
    "object_module_name": "numpy",
    "object_module_version": np.__version__,
}


_ARRAY_SAMPLE = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def pack_array() -> np.ndarray:
    return _ARRAY_SAMPLE


def validate_array(result: List[List[int]]) -> bool:
    return (np.array(result) == _ARRAY_SAMPLE).all()


def unpack_array(obj: np.ndarray):
    assert isinstance(obj, np.ndarray)
    assert (obj == _ARRAY_SAMPLE).all()


def prepare_array_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_array.{file_format}")
    formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
    formatter.save(obj=_ARRAY_SAMPLE, file_path=file_path)
    return file_path, temp_directory


class NumPyNDArrayPackagerTester(PackagerTester):
    """
    A tester for the `NumPyNDArrayPackager`.
    """

    PACKAGER_IN_TEST = NumPyNDArrayPackager

    TESTS = [
        PackTest(
            pack_handler="pack_array",
            log_hint="my_result",
            validation_function=validate_array,
            pack_parameters={},
            default_artifact_type_object=np.ones(1),
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_array_file,
                unpack_handler="unpack_array",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in NumPySupportedFormat.get_single_array_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_array",
            log_hint="my_result: result",
        ),
        PackToUnpackTest(
            pack_handler="pack_array",
            log_hint="my_result: object",
            expected_instructions=_COMMON_OBJECT_INSTRUCTIONS,
            unpack_handler="unpack_array",
        ),
        PackToUnpackTest(
            pack_handler="pack_array",
            log_hint="my_result: dataset",
            unpack_handler="unpack_array",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_array",
                log_hint={
                    "key": "my_array",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={"file_format": file_format},
                unpack_handler="unpack_array",
            )
            for file_format in NumPySupportedFormat.get_single_array_formats()
        ],
    ]


_NUMBER_SAMPLE = np.float64(5.10203)


def pack_number() -> np.number:
    return _NUMBER_SAMPLE


def validate_number(result: float) -> bool:
    return np.float64(result) == _NUMBER_SAMPLE


def unpack_number(obj: np.float64):
    assert isinstance(obj, np.float64)
    assert obj == _NUMBER_SAMPLE


class NumPyNumberPackagerTester(PackagerTester):
    """
    A tester for the `NumPyNumberPackager`.
    """

    PACKAGER_IN_TEST = NumPyNumberPackager

    TESTS = [
        PackTest(
            pack_handler="pack_number",
            log_hint="my_result",
            validation_function=validate_number,
        ),
        PackToUnpackTest(
            pack_handler="pack_number",
            log_hint="my_result",
        ),
        PackToUnpackTest(
            pack_handler="pack_number",
            log_hint="my_result: object",
            expected_instructions=_COMMON_OBJECT_INSTRUCTIONS,
            unpack_handler="unpack_number",
        ),
    ]


_ARRAY_DICT_SAMPLE = {f"my_array_{i}": _ARRAY_SAMPLE * i for i in range(1, 5)}


def pack_array_dict() -> Dict[str, np.ndarray]:
    return _ARRAY_DICT_SAMPLE


def unpack_array_dict(obj: Dict[str, np.ndarray]):
    assert isinstance(obj, dict) and all(
        isinstance(key, str) and isinstance(value, np.ndarray)
        for key, value in obj.items()
    )
    assert obj.keys() == _ARRAY_DICT_SAMPLE.keys()
    for obj_array, sample_array in zip(obj.values(), _ARRAY_DICT_SAMPLE.values()):
        assert (obj_array == sample_array).all()


def validate_array_dict(result: Dict[str, list]) -> bool:
    # Numppy arrays are serialized as lists:
    for key in _ARRAY_DICT_SAMPLE:
        array = result.pop(key)
        if not (np.array(array) == _ARRAY_DICT_SAMPLE[key]).all():
            return False
    return len(result) == 0


def prepare_array_dict_file(file_format: str, **save_kwargs) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
    formatter.save(obj=_ARRAY_DICT_SAMPLE, file_path=file_path, **save_kwargs)
    return file_path, temp_directory


class NumPyNDArrayDictPackagerTester(PackagerTester):
    """
    A tester for the `NumPyNDArrayDictPackager`.
    """

    PACKAGER_IN_TEST = NumPyNDArrayDictPackager

    TESTS = [
        PackTest(
            pack_handler="pack_array_dict",
            log_hint="my_result: result",
            validation_function=validate_array_dict,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_array_dict_file,
                unpack_handler="unpack_array_dict",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in NumPySupportedFormat.get_multi_array_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_array_dict",
            log_hint="my_array: result",
        ),
        PackToUnpackTest(
            pack_handler="pack_array_dict",
            log_hint="my_array: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": dict.__module__,
            },
            unpack_handler="unpack_array_dict",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_array_dict",
                log_hint={
                    "key": "my_array",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_array_dict",
            )
            for file_format in NumPySupportedFormat.get_multi_array_formats()
        ],
    ]


_ARRAY_LIST_SAMPLE = list(_ARRAY_DICT_SAMPLE.values())


def pack_array_list() -> List[np.ndarray]:
    return _ARRAY_LIST_SAMPLE


def unpack_array_list(obj: List[np.ndarray]):
    assert isinstance(obj, list) and all(isinstance(value, np.ndarray) for value in obj)
    for obj_array, sample_array in zip(obj, _ARRAY_LIST_SAMPLE):
        assert (obj_array == sample_array).all()


def validate_array_list(result: List[list]) -> bool:
    # Numppy arrays are serialized as lists:
    for result_array, sample_array in zip(result, _ARRAY_LIST_SAMPLE):
        if not (np.array(result_array) == sample_array).all():
            return False
    return True


def prepare_array_list_file(file_format: str, **save_kwargs) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
    formatter.save(obj=_ARRAY_LIST_SAMPLE, file_path=file_path, **save_kwargs)
    return file_path, temp_directory


class NumPyNDArrayListPackagerTester(PackagerTester):
    """
    A tester for the `NumPyNDArrayListPackager`.
    """

    PACKAGER_IN_TEST = NumPyNDArrayListPackager

    TESTS = [
        PackTest(
            pack_handler="pack_array_list",
            log_hint="my_result: result",
            validation_function=validate_array_list,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_array_list_file,
                unpack_handler="unpack_array_list",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in NumPySupportedFormat.get_multi_array_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_array_list",
            log_hint="my_array: result",
        ),
        PackToUnpackTest(
            pack_handler="pack_array_list",
            log_hint="my_array: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": dict.__module__,
            },
            unpack_handler="unpack_array_list",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_array_list",
                log_hint={
                    "key": "my_array",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_array_list",
            )
            for file_format in NumPySupportedFormat.get_multi_array_formats()
        ],
    ]
