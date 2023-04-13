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
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Tuple, Type, Union

import pytest

from mlrun import DataItem
from mlrun.artifacts import Artifact
from mlrun.errors import MLRunInvalidArgumentError, MLRunPackagePackagerCollectionError
from mlrun.package import DefaultPackager, Packager, PackagersManager


class PackagerA(Packager):
    """
    A simple packager to pack strings as results.
    """

    PACKABLE_OBJECT_TYPE = str

    @classmethod
    def get_default_artifact_type(cls, obj: Any) -> str:
        return "result"

    @classmethod
    def get_supported_artifact_types(cls) -> List[str]:
        return ["result"]

    @classmethod
    def is_packable(cls, object_type: Type, artifact_type: str = None) -> bool:
        return object_type is cls.PACKABLE_OBJECT_TYPE and artifact_type == "result"

    @classmethod
    def pack(
        cls, obj: str, artifact_type: Union[str, None], configurations: dict
    ) -> dict:
        return {f"{configurations['key']}_from_PackagerA": obj}

    @classmethod
    def unpack(
        cls,
        data_item: DataItem,
        artifact_type: Union[str, None],
        instructions: dict,
    ) -> str:
        pass


class PackagerB(DefaultPackager):
    """
    A default packager for strings. The artifact types "b1" and "b2" will be used to verify the future clear feature.
    """

    PACKABLE_OBJECT_TYPE = str

    @classmethod
    def pack_result(cls, obj: Any, key: str) -> dict:
        return {f"{key}_from_PackagerB": obj}

    @classmethod
    def pack_b1(
        cls,
        obj: str,
        key: str,
        fmt: str,
    ) -> Tuple[Artifact, dict]:
        # Create a temp directory:
        path = tempfile.mkdtemp()

        # Create a file:
        file_path = os.path.join(path, f"{key}.{fmt}")
        with open(file_path, "w") as file:
            file.write(obj)

        # Note for clearance:
        cls.future_clear(path=file_path)

        return Artifact(key=key, src_path=file_path), {"temp_dir": path}

    @classmethod
    def pack_b2(
        cls,
        obj: str,
        key: str,
        amount_of_files: int,
    ) -> Tuple[Artifact, dict]:
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
        cls.future_clear(path=path)

        return Artifact(key=key, src_path=zip_path), {
            "temp_dir": path,
            "amount_of_files": amount_of_files,
        }

    @classmethod
    def unpack_b1(cls, data_item: DataItem):
        pass

    @classmethod
    def unpack_b2(cls, data_item: DataItem, length: int):
        pass


class PackagerC(PackagerA):
    """
    Another packager to test collecting an inherited class of `Packager`. In addition, it is used to test the arbitrary
    log hint keys.
    """

    PACKABLE_OBJECT_TYPE = float

    @classmethod
    def pack(
        cls, obj: float, artifact_type: Union[str, None], configurations: dict
    ) -> dict:
        return {configurations["key"]: round(obj, configurations["n_round"])}


class NotAPackager:
    """
    Simple class to test an exception will be raised when trying to collect it.
    """

    pass


@pytest.mark.parametrize(
    "test_tuple",
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
    test_tuple: Tuple[List[str], Union[List[Type[Packager]], str]]
):
    """
    Test the manager's `collect_packagers` method.

    :param test_tuple: A tuple of the packagers to collect and the packager classes that should have been collected.
                       A string means an error should be raised.
    """
    # Prepare the test:
    packagers_to_collect, validation = test_tuple
    packagers_manager = PackagersManager()

    # Try to collect the packagers:
    try:
        packagers_manager.collect_packagers(packagers=packagers_to_collect)
    except MLRunPackagePackagerCollectionError as error:
        # Catch only if the validation is a string, otherwise it is a legitimate exception:
        if isinstance(validation, str):
            # Make sure the correct error was raised:
            assert validation in str(error)
            return
        raise error

    # Validate only the required packagers were collected:
    for packager in validation:
        assert packager in packagers_manager._packagers


@pytest.mark.parametrize(
    "test_tuple",
    [
        ([PackagerA, PackagerB], "_from_PackagerB"),
        ([PackagerB, PackagerA], "_from_PackagerA"),
    ],
)
def test_packagers_priority(test_tuple: Tuple[List[Type[Packager]], str]):
    """
    Test the priority of the collected packagers (last collected - highest priority).

    :param test_tuple: A tuple of the packagers to collect and the suffix the result key should have if it was
                       collected by the right packager.
    """
    # Prepare the test:
    packagers_to_collect, result_key_suffix = test_tuple
    packagers_manager = PackagersManager()
    packagers_manager.collect_packagers(packagers=packagers_to_collect)

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
    a_temp_dir = packagers_manager.artifacts[0].spec.packaging_instructions[
        "instructions"
    ]["temp_dir"]
    a_file = os.path.join(a_temp_dir, "a.txt")
    b_temp_dir = packagers_manager.artifacts[1].spec.packaging_instructions[
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
    "test_tuple",
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
    test_tuple: Tuple[
        str, Union[List[float], Dict[str, float]], Union[Dict[str, float], str]
    ]
):
    """
    Test the arbitrary log hint key prefixes "*" and "**".

    :param test_tuple: A tuple of the key to use in the log hint, the object to pack and the expected results that
                       should be packed. A string means an error should be raised.
    """
    # Prepare the test:
    key, obj, expected_results = test_tuple
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
