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
import ast
import os
import pathlib
import tempfile
from typing import Tuple

from mlrun import MLClientCtx
from mlrun.package.packagers.python_standard_library_packagers import (
    BoolPackager,
    BytearrayPackager,
    BytesPackager,
    DictPackager,
    FloatPackager,
    FrozensetPackager,
    IntPackager,
    ListPackager,
    PathPackager,
    SetPackager,
    StrPackager,
    TuplePackager,
)
from mlrun.package.utils import ArchiveSupportedFormat, StructFileSupportedFormat
from tests.package.packager_tester import (
    COMMON_OBJECT_INSTRUCTIONS,
    PackagerTester,
    PackTest,
    PackToUnpackTest,
    UnpackTest,
)

# ----------------------------------------------------------------------------------------------------------------------
# builtins packagers:
# ----------------------------------------------------------------------------------------------------------------------

_INT_SAMPLE = 7


def pack_int() -> int:
    return _INT_SAMPLE


def validate_int(result: int) -> bool:
    return result == _INT_SAMPLE


def unpack_int(obj: int):
    assert isinstance(obj, int)
    assert obj == _INT_SAMPLE


class IntPackagerTester(PackagerTester):
    """
    A tester for the `IntPackager`.
    """

    PACKAGER_IN_TEST = IntPackager

    TESTS = [
        PackTest(
            pack_handler="pack_int",
            log_hint="my_result",
            validation_function=validate_int,
        ),
        PackToUnpackTest(
            pack_handler="pack_int",
            log_hint="my_result",
        ),
        PackToUnpackTest(
            pack_handler="pack_int",
            log_hint="my_result: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": int.__module__,
            },
            unpack_handler="unpack_int",
        ),
    ]


_FLOAT_SAMPLE = 0.97123


def pack_float() -> float:
    return _FLOAT_SAMPLE


def validate_float(result: float) -> bool:
    return result == _FLOAT_SAMPLE


def unpack_float(obj: float):
    assert isinstance(obj, float)
    assert obj == _FLOAT_SAMPLE


class FloatPackagerTester(PackagerTester):
    """
    A tester for the `FloatPackager`.
    """

    PACKAGER_IN_TEST = FloatPackager

    TESTS = [
        PackTest(
            pack_handler="pack_float",
            log_hint="my_result",
            validation_function=validate_float,
        ),
        PackToUnpackTest(
            pack_handler="pack_float",
            log_hint="my_result",
        ),
        PackToUnpackTest(
            pack_handler="pack_float",
            log_hint="my_result: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": float.__module__,
            },
            unpack_handler="unpack_float",
        ),
    ]


_BOOL_SAMPLE = True


def pack_bool() -> float:
    return _BOOL_SAMPLE


def validate_bool(result: bool) -> bool:
    return result is _BOOL_SAMPLE


def unpack_bool(obj: bool):
    assert isinstance(obj, bool)
    assert obj is _BOOL_SAMPLE


class BoolPackagerTester(PackagerTester):
    """
    A tester for the `BoolPackager`.
    """

    PACKAGER_IN_TEST = BoolPackager

    TESTS = [
        PackTest(
            pack_handler="pack_bool",
            log_hint="my_result",
            validation_function=validate_bool,
        ),
        PackToUnpackTest(
            pack_handler="pack_bool",
            log_hint="my_result",
        ),
        PackToUnpackTest(
            pack_handler="pack_bool",
            log_hint="my_result: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": bool.__module__,
            },
            unpack_handler="unpack_bool",
        ),
    ]


_STR_RESULT_SAMPLE = "I'm a string."
_STR_FILE_SAMPLE = "Something written in a file..."
_STR_DIRECTORY_FILES_SAMPLE = "I'm text file number {}"


def pack_str() -> str:
    return _STR_RESULT_SAMPLE


def pack_str_path_file(context: MLClientCtx) -> str:
    file_path = os.path.join(context.artifact_path, "my_file.txt")
    with open(file_path, "w") as file:
        file.write(_STR_FILE_SAMPLE)
    return file_path


def pack_str_path_directory(context: MLClientCtx) -> str:
    directory_path = os.path.join(context.artifact_path, "my_directory")
    os.makedirs(directory_path)
    for i in range(5):
        with open(os.path.join(directory_path, f"file_{i}.txt"), "w") as file:
            file.write(_STR_DIRECTORY_FILES_SAMPLE.format(i))
    return directory_path


def validate_str_result(result: str) -> bool:
    return result == _STR_RESULT_SAMPLE


def unpack_str(obj: str):
    assert isinstance(obj, str)
    assert obj == _STR_RESULT_SAMPLE


def unpack_str_path_file(obj: str):
    assert isinstance(obj, str)
    with open(obj, "r") as file:
        file_content = file.read()
    assert file_content == _STR_FILE_SAMPLE


def unpack_str_path_directory(obj: str):
    assert isinstance(obj, str)
    for i in range(5):
        with open(os.path.join(obj, f"file_{i}.txt"), "r") as file:
            file_content = file.read()
        assert file_content == _STR_DIRECTORY_FILES_SAMPLE.format(i)


def prepare_str_path_file() -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, "my_file.txt")
    with open(file_path, "w") as file:
        file.write(_STR_FILE_SAMPLE)
    return file_path, temp_directory


class StrPackagerTester(PackagerTester):
    """
    A tester for the `StrPackager`.
    """

    PACKAGER_IN_TEST = StrPackager

    TESTS = [
        PackTest(
            pack_handler="pack_str",
            log_hint="my_result",
            validation_function=validate_str_result,
            pack_parameters={},
        ),
        UnpackTest(
            prepare_input_function=prepare_str_path_file,
            unpack_handler="unpack_str_path_file",
        ),
        PackToUnpackTest(
            pack_handler="pack_str",
            log_hint="my_result",
        ),
        PackToUnpackTest(
            pack_handler="pack_str",
            log_hint="my_result: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": str.__module__,
            },
            unpack_handler="unpack_str",
        ),
        PackToUnpackTest(
            pack_handler="pack_str_path_file",
            log_hint="my_file: path",
            expected_instructions={"is_directory": False},
            unpack_handler="unpack_str_path_file",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_str_path_directory",
                log_hint={
                    "key": "my_dir",
                    "artifact_type": "path",
                    "archive_format": archive_format,
                },
                expected_instructions={
                    "is_directory": True,
                    "archive_format": archive_format,
                },
                unpack_handler="unpack_str_path_directory",
            )
            for archive_format in ArchiveSupportedFormat.get_all_formats()
        ],
    ]


_DICT_SAMPLE = {"a1": {"a2": [1, 2, 3], "b2": [4, 5, 6]}, "b1": {"b2": [4, 5, 6]}}


def pack_dict() -> dict:
    return _DICT_SAMPLE


def unpack_dict(obj: dict):
    assert isinstance(obj, dict)
    assert obj == _DICT_SAMPLE


def validate_dict_result(result: dict) -> bool:
    return result == _DICT_SAMPLE


def prepare_dict_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=_DICT_SAMPLE, file_path=file_path)
    return file_path, temp_directory


class DictPackagerTester(PackagerTester):
    """
    A tester for the `DictPackager`.
    """

    PACKAGER_IN_TEST = DictPackager

    TESTS = [
        PackTest(
            pack_handler="pack_dict",
            log_hint="my_dict",
            validation_function=validate_dict_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_dict_file,
                unpack_handler="unpack_dict",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_dict",
            log_hint="my_dict",
        ),
        PackToUnpackTest(
            pack_handler="pack_dict",
            log_hint="my_dict: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": dict.__module__,
            },
            unpack_handler="unpack_dict",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_dict",
                log_hint={
                    "key": "my_dict",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_dict",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


_LIST_SAMPLE = [1, 2, 3, {"a": 1, "b": 2}]


def pack_list() -> list:
    return _LIST_SAMPLE


def unpack_list(obj: list):
    assert isinstance(obj, list)
    assert obj == _LIST_SAMPLE


def validate_list_result(result: list) -> bool:
    return result == _LIST_SAMPLE


def prepare_list_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=_LIST_SAMPLE, file_path=file_path)
    return file_path, temp_directory


class ListPackagerTester(PackagerTester):
    """
    A tester for the `ListPackager`.
    """

    PACKAGER_IN_TEST = ListPackager

    TESTS = [
        PackTest(
            pack_handler="pack_list",
            log_hint="my_list",
            validation_function=validate_list_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_list_file,
                unpack_handler="unpack_list",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_list",
            log_hint="my_list",
        ),
        PackToUnpackTest(
            pack_handler="pack_list",
            log_hint="my_list: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": tuple.__module__,
            },
            unpack_handler="unpack_list",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_list",
                log_hint={
                    "key": "my_list",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_list",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


_TUPLE_SAMPLE = (1, 2, 3)


def pack_tuple() -> tuple:
    return _TUPLE_SAMPLE


def unpack_tuple(obj: tuple):
    assert isinstance(obj, tuple)
    assert obj == _TUPLE_SAMPLE


def validate_tuple_result(result: list) -> bool:
    # Tuples are serialized as lists:
    return tuple(result) == _TUPLE_SAMPLE


def prepare_tuple_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=list(_TUPLE_SAMPLE), file_path=file_path)
    return file_path, temp_directory


class TuplePackagerTester(PackagerTester):
    """
    A tester for the `TuplePackager`.
    """

    PACKAGER_IN_TEST = TuplePackager

    TESTS = [
        PackTest(
            pack_handler="pack_tuple",
            log_hint="my_tuple",
            validation_function=validate_tuple_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_tuple_file,
                unpack_handler="unpack_tuple",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_tuple",
            log_hint="my_tuple",
        ),
        PackToUnpackTest(
            pack_handler="pack_tuple",
            log_hint="my_tuple: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": tuple.__module__,
            },
            unpack_handler="unpack_tuple",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_tuple",
                log_hint={
                    "key": "my_tuple",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_tuple",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


_SET_SAMPLE = {1, 2, 3}


def pack_set() -> set:
    return _SET_SAMPLE


def unpack_set(obj: set):
    assert isinstance(obj, set)
    assert obj == _SET_SAMPLE


def validate_set_result(result: list) -> bool:
    # Sets are serialized as lists:
    return set(result) == _SET_SAMPLE


def prepare_set_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=list(_SET_SAMPLE), file_path=file_path)
    return file_path, temp_directory


class SetPackagerTester(PackagerTester):
    """
    A tester for the `SetPackager`.
    """

    PACKAGER_IN_TEST = SetPackager

    TESTS = [
        PackTest(
            pack_handler="pack_set",
            log_hint="my_set",
            validation_function=validate_set_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_set_file,
                unpack_handler="unpack_set",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_set",
            log_hint="my_set",
        ),
        PackToUnpackTest(
            pack_handler="pack_set",
            log_hint="my_set: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": set.__module__,
            },
            unpack_handler="unpack_set",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_set",
                log_hint={
                    "key": "my_set",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_set",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


_FROZENSET_SAMPLE = frozenset([1, 2, 3])


def pack_frozenset() -> frozenset:
    return _FROZENSET_SAMPLE


def unpack_frozenset(obj: frozenset):
    assert isinstance(obj, frozenset)
    assert obj == _FROZENSET_SAMPLE


def validate_frozenset_result(result: list) -> bool:
    # Frozen sets are serialized as lists:
    return frozenset(result) == _FROZENSET_SAMPLE


def prepare_frozenset_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=list(_FROZENSET_SAMPLE), file_path=file_path)
    return file_path, temp_directory


class FrozensetPackagerTester(PackagerTester):
    """
    A tester for the `FrozensetPackager`.
    """

    PACKAGER_IN_TEST = FrozensetPackager

    TESTS = [
        PackTest(
            pack_handler="pack_frozenset",
            log_hint="my_frozenset",
            validation_function=validate_frozenset_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_frozenset_file,
                unpack_handler="unpack_frozenset",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_frozenset",
            log_hint="my_frozenset",
        ),
        PackToUnpackTest(
            pack_handler="pack_frozenset",
            log_hint="my_frozenset: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": set.__module__,
            },
            unpack_handler="unpack_frozenset",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_frozenset",
                log_hint={
                    "key": "my_frozenset",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_frozenset",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


_BYTEARRAY_SAMPLE = bytearray([1, 2, 3])


def pack_bytearray() -> bytearray:
    return _BYTEARRAY_SAMPLE


def unpack_bytearray(obj: bytearray):
    assert isinstance(obj, bytearray)
    assert obj == _BYTEARRAY_SAMPLE


def validate_bytearray_result(result: str) -> bool:
    # Byte arrays are serialized as strings (not decoded):
    return bytearray(ast.literal_eval(result)) == _BYTEARRAY_SAMPLE


def prepare_bytearray_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=list(_BYTEARRAY_SAMPLE), file_path=file_path)
    return file_path, temp_directory


class BytearrayPackagerTester(PackagerTester):
    """
    A tester for the `BytearrayPackager`.
    """

    PACKAGER_IN_TEST = BytearrayPackager

    TESTS = [
        PackTest(
            pack_handler="pack_bytearray",
            log_hint="my_bytearray",
            validation_function=validate_bytearray_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_bytearray_file,
                unpack_handler="unpack_bytearray",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_bytearray",
            log_hint="my_bytearray",
        ),
        PackToUnpackTest(
            pack_handler="pack_bytearray",
            log_hint="my_bytearray: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": set.__module__,
            },
            unpack_handler="unpack_bytearray",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_bytearray",
                log_hint={
                    "key": "my_bytearray",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_bytearray",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


_BYTES_SAMPLE = b"I'm a byte string."


def pack_bytes() -> bytes:
    return _BYTES_SAMPLE


def unpack_bytes(obj: bytes):
    assert isinstance(obj, bytes)
    assert obj == _BYTES_SAMPLE


def validate_bytes_result(result: str) -> bool:
    # Bytes are serialized as strings (not decoded):
    return ast.literal_eval(result) == _BYTES_SAMPLE


def prepare_bytes_file(file_format: str) -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    file_path = os.path.join(temp_directory, f"my_file.{file_format}")
    formatter = StructFileSupportedFormat.get_format_handler(fmt=file_format)
    formatter.write(obj=list(_BYTES_SAMPLE), file_path=file_path)
    return file_path, temp_directory


class BytesPackagerTester(PackagerTester):
    """
    A tester for the `BytesPackager`.
    """

    PACKAGER_IN_TEST = BytesPackager

    TESTS = [
        PackTest(
            pack_handler="pack_bytes",
            log_hint="my_bytes",
            validation_function=validate_bytes_result,
        ),
        *[
            UnpackTest(
                prepare_input_function=prepare_bytes_file,
                unpack_handler="unpack_bytes",
                prepare_parameters={"file_format": file_format},
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
        PackToUnpackTest(
            pack_handler="pack_bytes",
            log_hint="my_bytes",
        ),
        PackToUnpackTest(
            pack_handler="pack_bytes",
            log_hint="my_bytes: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": set.__module__,
            },
            unpack_handler="unpack_bytes",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_bytes",
                log_hint={
                    "key": "my_bytes",
                    "artifact_type": "file",
                    "file_format": file_format,
                },
                expected_instructions={
                    "file_format": file_format,
                },
                unpack_handler="unpack_bytes",
            )
            for file_format in StructFileSupportedFormat.get_all_formats()
        ],
    ]


# ----------------------------------------------------------------------------------------------------------------------
# pathlib packagers:
# ----------------------------------------------------------------------------------------------------------------------


_PATH_RESULT_SAMPLE = pathlib.Path("I'm a path.")


def pack_path() -> pathlib.Path:
    return _PATH_RESULT_SAMPLE


def pack_path_file(context: MLClientCtx) -> pathlib.Path:
    file_path = pathlib.Path(context.artifact_path) / "my_file.txt"
    with open(file_path, "w") as file:
        file.write(_STR_FILE_SAMPLE)
    return file_path


def pack_path_directory(context: MLClientCtx) -> pathlib.Path:
    directory_path = pathlib.Path(context.artifact_path) / "my_directory"
    os.makedirs(directory_path)
    for i in range(5):
        with open(directory_path / f"file_{i}.txt", "w") as file:
            file.write(_STR_DIRECTORY_FILES_SAMPLE.format(i))
    return directory_path


def validate_path_result(result: pathlib.Path) -> bool:
    return pathlib.Path(result) == _PATH_RESULT_SAMPLE


def unpack_path(obj: pathlib.Path):
    assert isinstance(obj, pathlib.Path)
    assert obj == _PATH_RESULT_SAMPLE


def unpack_path_file(obj: pathlib.Path):
    assert isinstance(obj, pathlib.Path)
    with open(obj, "r") as file:
        file_content = file.read()
    assert file_content == _STR_FILE_SAMPLE


def unpack_path_directory(obj: pathlib.Path):
    assert isinstance(obj, pathlib.Path)
    for i in range(5):
        with open(obj / f"file_{i}.txt", "r") as file:
            file_content = file.read()
        assert file_content == _STR_DIRECTORY_FILES_SAMPLE.format(i)


class PathPackagerTester(PackagerTester):
    """
    A tester for the `PathPackager`.
    """

    PACKAGER_IN_TEST = PathPackager

    TESTS = [
        PackTest(
            pack_handler="pack_path",
            log_hint="my_result: result",
            validation_function=validate_path_result,
            pack_parameters={},
        ),
        UnpackTest(
            prepare_input_function=prepare_str_path_file,  # Using str preparing method - same thing
            unpack_handler="unpack_path_file",
        ),
        PackToUnpackTest(
            pack_handler="pack_path",
            log_hint="my_result: result",
        ),
        PackToUnpackTest(
            pack_handler="pack_path",
            log_hint="my_result: object",
            expected_instructions={
                **COMMON_OBJECT_INSTRUCTIONS,
                "object_module_name": pathlib.Path.__module__,
            },
            unpack_handler="unpack_path",
        ),
        PackToUnpackTest(
            pack_handler="pack_path_file",
            log_hint="my_file",
            expected_instructions={"is_directory": False},
            unpack_handler="unpack_path_file",
        ),
        *[
            PackToUnpackTest(
                pack_handler="pack_path_directory",
                log_hint={
                    "key": "my_dir",
                    "archive_format": archive_format,
                },
                expected_instructions={
                    "is_directory": True,
                    "archive_format": archive_format,
                },
                unpack_handler="unpack_path_directory",
            )
            for archive_format in ArchiveSupportedFormat.get_all_formats()
        ],
    ]
