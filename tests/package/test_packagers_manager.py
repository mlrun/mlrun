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
import shutil
import tempfile
import zipfile
from typing import Any, Optional, Union

import pytest

from mlrun import DataItem
from mlrun.artifacts import Artifact
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package import (
    DefaultPackager,
    MLRunPackageCollectionError,
    MLRunPackageUnpackingError,
    Packager,
    PackagersManager,
)


class PackagerA(Packager):
    """
    A simple packager to pack strings as results.
    """

    PACKABLE_OBJECT_TYPE = str

    def get_default_packing_artifact_type(self, obj: Any) -> str:
        return "result"

    def get_default_unpacking_artifact_type(self, data_item: DataItem) -> str:
        return "result"

    def get_supported_artifact_types(self) -> list[str]:
        return ["result"]

    def is_packable(
        self,
        obj: Any,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> bool:
        return type(obj) is self.PACKABLE_OBJECT_TYPE and artifact_type == "result"

    def pack(
        self,
        obj: str,
        key: Optional[str] = None,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> dict:
        return {f"{key}_from_PackagerA": obj}

    def unpack(
        self,
        data_item: DataItem,
        artifact_type: Optional[str] = None,
        instructions: Optional[dict] = None,
    ) -> str:
        pass


class PackagerB(DefaultPackager):
    """
    A default packager for strings. The artifact types "b1" and "b2" will be used to verify the future clear feature.
    """

    PACKABLE_OBJECT_TYPE = str
    DEFAULT_PACKING_ARTIFACT_TYPE = "b1"
    DEFAULT_UNPACKING_ARTIFACT_TYPE = "b1"

    def pack_result(self, obj: Any, key: str) -> dict:
        return {f"{key}_from_PackagerB": obj}

    def pack_b1(
        self,
        obj: str,
        key: str,
        fmt: str,
    ) -> tuple[Artifact, dict]:
        # Create a temp directory:
        path = tempfile.mkdtemp()

        # Create a file:
        file_path = os.path.join(path, f"{key}.{fmt}")
        with open(file_path, "w") as file:
            file.write(obj)

        # Note for clearance:
        self.add_future_clearing_path(path=file_path)

        return Artifact(key=key, src_path=file_path), {"temp_dir": path}

    def pack_b2(
        self,
        obj: str,
        key: str,
        amount_of_files: int,
    ) -> tuple[Artifact, dict]:
        # Create a temp directory:
        path = tempfile.mkdtemp()

        # Create some files in it:
        files = []
        for i in range(amount_of_files):
            file_path = os.path.join(path, f"{i}.txt")
            files.append(file_path)
            with open(file_path, "w") as file:
                file.write(obj)

        # Zip them:
        zip_path = os.path.join(path, f"{key}.zip")
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for txt_file_path in files:
                zip_file.write(txt_file_path)

        # Note for clearance:
        self.add_future_clearing_path(path=path)

        return Artifact(key=key, src_path=zip_path), {
            "temp_dir": path,
            "amount_of_files": amount_of_files,
        }

    def unpack_b1(self, data_item: DataItem):
        pass

    def unpack_b2(self, data_item: DataItem, length: int):
        pass


class PackagerC(PackagerA):
    """
    Another packager to test collecting an inherited class of `Packager`. In addition, it is used to test the arbitrary
    log hint keys.
    """

    PACKABLE_OBJECT_TYPE = float

    def pack(
        self,
        obj: float,
        key: Optional[str] = None,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> dict:
        return {key: round(obj, configurations["n_round"])}

    def unpack(
        self,
        data_item: DataItem,
        artifact_type: Optional[str] = None,
        instructions: Optional[dict] = None,
    ) -> float:
        return data_item.key * 2


class NotAPackager:
    """
    Simple class to test an exception will be raised when trying to collect it.
    """

    pass


@pytest.mark.parametrize(
    "packagers_to_collect, validation",
    [
        (["tests.package.test_packagers_manager.PackagerA"], [PackagerA]),
        (
            [
                "tests.package.test_packagers_manager.PackagerA",
                "tests.package.test_packagers_manager.PackagerC",
            ],
            [PackagerA, PackagerC],
        ),
        (
            ["tests.package.test_packagers_manager.*"],
            [PackagerA, PackagerB, PackagerC],
        ),
        (
            ["tests.package.module_not_exist.PackagerA"],
            "The packager 'PackagerA' could not be collected from the module 'tests.package.module_not_exist'",
        ),
        (
            ["tests.package.test_packagers_manager.PackagerNotExist"],
            "The packager 'PackagerNotExist' could not be collected as it does not exist in the module",
        ),
        (
            ["tests.package.test_packagers_manager.NotAPackager"],
            "The packager 'NotAPackager' could not be collected as it is not a `mlrun.Packager`",
        ),
    ],
)
def test_collect_packagers(
    packagers_to_collect: list[str], validation: Union[list[type[Packager]], str]
):
    """
    Test the manager's `collect_packagers` method.

    :param packagers_to_collect: The packagers to collect.
    :param validation:           The packager classes that should have been collected. A string means an error should
                                 be raised.
    """
    # Prepare the test:
    packagers_manager = PackagersManager()

    # Try to collect the packagers:
    try:
        packagers_manager.collect_packagers(packagers=packagers_to_collect)
    except MLRunPackageCollectionError as error:
        # Catch only if the validation is a string, otherwise it is a legitimate exception:
        if isinstance(validation, str):
            # Make sure the correct error was raised:
            assert validation in str(error)
            return
        raise error

    # Validate only the required packagers were collected:
    assert set(
        packager.__class__.__name__ for packager in packagers_manager._packagers
    ) == set(packager.__name__ for packager in validation)


@pytest.mark.parametrize(
    "packagers_to_collect, result_key_suffix",
    [
        ([PackagerA, PackagerB], "_from_PackagerB"),
        ([PackagerB, PackagerA], "_from_PackagerA"),
    ],
)
@pytest.mark.parametrize("set_via_default_priority", [True, False])
def test_packagers_priority(
    packagers_to_collect: list[type[Packager]],
    result_key_suffix: str,
    set_via_default_priority: bool,
):
    """
    Test the priority of the collected packagers (last collected will be set with the highest priority).

    :param packagers_to_collect:     The packagers to collect
    :param result_key_suffix:        The suffix the result key should have if it was collected by the right packager.
    :param set_via_default_priority: Whether to set the priority via the class or the default priority in collection.
    """
    # Reset priorities (when performing multiple runs the class priority is remained set from previous run):
    PackagerA.PRIORITY = ...
    PackagerB.PRIORITY = ...

    # Collect the packagers:
    packagers_manager = PackagersManager()
    for packager, priority in zip(packagers_to_collect, [2, 1]):
        if not set_via_default_priority:
            packager.PRIORITY = priority
        packagers_manager.collect_packagers(
            packagers=[packager], default_priority=priority
        )
        for collected_packager in packagers_manager._packagers:
            if collected_packager.__class__.__name__ == packager:
                assert collected_packager.priority == priority

    # Pack a string as a result:
    key = "some_key"
    packagers_manager.pack(
        obj="some string", log_hint={"key": key, "artifact_type": "result"}
    )

    # Make sure the correct packager packed the result by the suffix:
    assert f"{key}{result_key_suffix}" in packagers_manager.results


def test_clear_packagers_outputs():
    """
    Test the manager's `clear_packagers_outputs` method.
    """
    # Prepare the test:
    packagers_manager = PackagersManager()
    packagers_manager.collect_packagers(packagers=[PackagerB])

    # Pack objects that will create temporary files and directories:
    packagers_manager.pack(
        obj="I'm a test.",
        log_hint={"key": "a", "artifact_type": "b1", "fmt": "txt"},
    )
    packagers_manager.pack(
        obj="I'm another test.",
        log_hint={
            "key": "b",
            "artifact_type": "b2",
            "amount_of_files": 3,
        },
    )

    # Get the created files:
    a_temp_dir = packagers_manager.artifacts[0].spec.unpackaging_instructions[
        "instructions"
    ]["temp_dir"]
    a_file = os.path.join(a_temp_dir, "a.txt")
    b_temp_dir = packagers_manager.artifacts[1].spec.unpackaging_instructions[
        "instructions"
    ]["temp_dir"]

    # Assert they do exist before clearing up:
    assert os.path.exists(a_file)
    assert os.path.exists(b_temp_dir)

    # Clear:
    packagers_manager.clear_packagers_outputs()

    # Assert the clearance:
    assert not os.path.exists(a_file)
    assert not os.path.exists(b_temp_dir)

    # Remove remained directory (we tested the clearance of a file and a directory, so we need to delete the directory
    # of the cleared file (it's directory was not marked as future clear)):
    shutil.rmtree(a_temp_dir)


@pytest.mark.parametrize(
    "key, obj, expected_results",
    [
        (
            "*list_",
            [0.12111, 0.56111],
            {"list_0": 0.12, "list_1": 0.56},
        ),
        (
            "*set_",
            {0.12111, 0.56111},
            {"set_0": 0.12, "set_1": 0.56},
        ),
        (
            "*",
            (0.12111, 0.56111),
            {"0": 0.12, "1": 0.56},
        ),
        (
            "*error",
            0.12111,
            "The log hint key '*error' has an iterable unpacking prefix ('*')",
        ),
        (
            "**dict_",
            {"a": 0.12111, "b": 0.56111},
            {"dict_a": 0.12, "dict_b": 0.56},
        ),
        ("**", {"a": 0.12111, "b": 0.56111}, {"a": 0.12, "b": 0.56}),
        (
            "**error",
            0.12111,
            "The log hint key '**error' has a dictionary unpacking prefix ('**')",
        ),
    ],
)
def test_arbitrary_log_hint(
    key: str,
    obj: Union[list, dict, tuple, set],
    expected_results: Union[dict[str, float], str],
):
    """
    Test the arbitrary log hint key prefixes "*" and "**".

    :param key:              The key to use in the log hint
    :param obj:              The object to pack
    :param expected_results: The expected results that should be packed. A string means an error should be raised.
    """
    # Prepare the test:
    packagers_manager = PackagersManager()
    packagers_manager.collect_packagers(packagers=[PackagerC])

    # Pack an arbitrary amount of objects:
    try:
        packagers_manager.pack(
            obj=obj, log_hint={"key": key, "artifact_type": "result", "n_round": 2}
        )
    except MLRunInvalidArgumentError as error:
        # Catch only if the expected results is a string, otherwise it is a legitimate exception:
        if isinstance(expected_results, str):
            assert expected_results in str(error)
            return
        raise error

    # Validate multiple packages were packed:
    assert packagers_manager.results == expected_results


class _DummyDataItem:
    def __init__(self, key: str, is_artifact: bool = False):
        self.key = key
        self.artifact_url = ""
        self._is_artifact = is_artifact

    def get_artifact_type(self) -> bool:
        return self._is_artifact


@pytest.mark.parametrize(
    "data, type_hint, expected_results",
    [
        (
            0.5,
            Union[int, bytes, float, int],
            1.0,
        ),
        (
            0.5,
            Union[int, bytes, int],
            "Could not unpack data item with the hinted type",
        ),
    ],
)
def test_plural_type_hint_unpacking(
    data: Any,
    type_hint: Any,
    expected_results: Union[Any, str],
):
    """
    Test unpacking when plural type hint is given (for example: a union of types).

    :param data:             The data of the data item to unpack.
    :param type_hint:        The plural type hint of ths data item.
    :param expected_results: The expected results that should be unpacked. A string means an error should be raised.
    """
    # Prepare the test:
    packagers_manager = PackagersManager()
    packagers_manager.collect_packagers(packagers=[PackagerC])

    # Pack an arbitrary amount of objects:
    try:
        value = packagers_manager.unpack(
            data_item=_DummyDataItem(key=data), type_hint=type_hint
        )
    except MLRunPackageUnpackingError as error:
        # Catch only if the expected results is a string, otherwise it is a legitimate exception:
        if isinstance(expected_results, str):
            assert expected_results in str(error)
            return
        raise error

    # Validate multiple packages were packed:
    assert value == expected_results
