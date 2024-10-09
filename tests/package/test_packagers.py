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
import inspect
import shutil
import tempfile
import typing
from typing import Union

import pytest

import mlrun
from mlrun.package import ArtifactType, LogHintKey, PackagersManager
from mlrun.package.utils import LogHintUtils
from mlrun.runtimes import KubejobRuntime

from .packager_tester import PackagerTester, PackTest, PackToUnpackTest, UnpackTest
from .packagers_testers.default_packager_tester import DefaultPackagerTester
from .packagers_testers.numpy_packagers_testers import (
    NumPyNDArrayDictPackagerTester,
    NumPyNDArrayListPackagerTester,
    NumPyNDArrayPackagerTester,
    NumPyNumberPackagerTester,
)
from .packagers_testers.pandas_packagers_testers import (
    PandasDataFramePackagerTester,
    PandasSeriesPackagerTester,
)
from .packagers_testers.python_standard_library_packagers_testers import (
    BoolPackagerTester,
    BytearrayPackagerTester,
    BytesPackagerTester,
    DictPackagerTester,
    FloatPackagerTester,
    FrozensetPackagerTester,
    IntPackagerTester,
    ListPackagerTester,
    NonePackagerTester,
    PathPackagerTester,
    SetPackagerTester,
    StrPackagerTester,
    TuplePackagerTester,
)
from .packagers_testers.sklearn_packager_tester import SklearnPackagerTester

# All the testers to be included in the tests:
_PACKAGERS_TESTERS = [
    # DefaultPackagerTester,
    # NonePackagerTester,
    # BoolPackagerTester,
    # BytearrayPackagerTester,
    # BytesPackagerTester,
    # DictPackagerTester,
    # FloatPackagerTester,
    # FrozensetPackagerTester,
    # IntPackagerTester,
    # ListPackagerTester,
    # SetPackagerTester,
    # StrPackagerTester,
    # TuplePackagerTester,
    # PathPackagerTester,
    # NumPyNDArrayPackagerTester,
    # NumPyNumberPackagerTester,
    # NumPyNDArrayDictPackagerTester,
    # NumPyNDArrayListPackagerTester,
    # PandasDataFramePackagerTester,
    # PandasSeriesPackagerTester,
    SklearnPackagerTester,
]


def _get_tests_tuples(
    test_type: Union[type[PackTest], type[UnpackTest], type[PackToUnpackTest]],
) -> list[tuple[type[PackagerTester], PackTest]]:
    return [
        (tester, test)
        for tester in _PACKAGERS_TESTERS
        for test in tester.TESTS
        if isinstance(test, test_type)
    ]


def _setup_test(
    tester: type[PackagerTester],
    test: Union[PackTest, UnpackTest, PackToUnpackTest],
    test_directory: str,
) -> KubejobRuntime:
    # Enabled logging tuples only if the tuple test is about to be setup:
    if isinstance(test, (PackTest, PackToUnpackTest)) and tester is TuplePackagerTester:
        mlrun.mlconf.packagers.pack_tuples = True

    # Create a project for this tester:
    project = mlrun.get_or_create_project(
        name="default", context=test_directory, allow_cross_project=True
    )

    project.add_custom_packager(
        "mlrun.package.packagers.sklearn_packager.SklearnModelPack", is_mandatory=True
    )
    # Create a MLRun function using the tester source file (all the functions must be located in it):
    return project.set_function(
        func=inspect.getfile(tester),
        name=tester.__name__.lower(),
        kind="job",
        image="mlrun/mlrun",
    )


def _get_key_and_artifact_type(
    tester: type[PackagerTester], test: Union[PackTest, PackToUnpackTest]
) -> tuple[str, str]:
    # Parse the log hint (in case it is a string):
    log_hint = LogHintUtils.parse_log_hint(log_hint=test.log_hint)

    # Extract the key:
    key = log_hint[LogHintKey.KEY]

    # Get the artifact type (either from the log hint or from the packager - the default artifact type):
    artifact_type = (
        log_hint[LogHintKey.ARTIFACT_TYPE]
        if LogHintKey.ARTIFACT_TYPE in log_hint
        else tester.PACKAGER_IN_TEST.get_default_packing_artifact_type(
            obj=test.default_artifact_type_object
        )
    )

    return key, artifact_type


@pytest.mark.parametrize(
    "tester, test",
    _get_tests_tuples(test_type=PackTest),
)
def test_packager_pack(rundb_mock, tester: type[PackagerTester], test: PackTest):
    """
    Test a packager's packing.

    :param rundb_mock: A runDB mock fixture.
    :param tester: The `PackagerTester` class to get the functions to run from.
    :param test:   The `PackTest` tuple with the test parameters.
    """
    # Set up the test, creating a project and a MLRun function:
    test_directory = tempfile.TemporaryDirectory()
    mlrun_function = _setup_test(
        tester=tester, test=test, test_directory=test_directory.name
    )

    # Run the packing handler:
    try:
        pack_run = mlrun_function.run(
            name="pack",
            handler=test.pack_handler,
            params=test.pack_parameters,
            returns=[test.log_hint],
            artifact_path=test_directory.name,
            local=True,
        )

        # Verify the packaged output:
        key, artifact_type = _get_key_and_artifact_type(tester=tester, test=test)
        if artifact_type == ArtifactType.RESULT:
            assert key in pack_run.status.results
            assert test.validation_function(
                pack_run.status.results[key], **test.validation_parameters
            )
        else:
            assert key in pack_run.outputs
            assert test.validation_function(
                pack_run._artifact(key=key), **test.validation_parameters
            )
    except Exception as exception:
        # An error was raised, check if the test failed or should have failed:
        if test.exception is None:
            raise exception
        # Make sure the expected exception was raised:
        assert test.exception in str(exception)

    # Clear the tests outputs:
    test_directory.cleanup()


@pytest.mark.parametrize(
    "tester, test",
    _get_tests_tuples(test_type=UnpackTest),
)
def test_packager_unpack(rundb_mock, tester: type[PackagerTester], test: UnpackTest):
    """
    Test a packager's unpacking.

    :param rundb_mock: A runDB mock fixture.
    :param tester: The `PackagerTester` class to get the functions to run from.
    :param test:   The `UnpackTest` tuple with the test parameters.
    """
    # Create the input path to send for unpacking:
    input_path, temp_directory = test.prepare_input_function(**test.prepare_parameters)

    # Set up the test, creating a project and a MLRun function:
    test_directory = tempfile.TemporaryDirectory()
    mlrun_function = _setup_test(
        tester=tester, test=test, test_directory=test_directory.name
    )

    # Run the packing handler:
    try:
        mlrun_function.run(
            name="unpack",
            handler=test.unpack_handler,
            inputs={"obj": input_path},
            params=test.unpack_parameters,
            artifact_path=test_directory.name,
            local=True,
        )
    except Exception as exception:
        # An error was raised, check if the test failed or should have failed:
        if test.exception is None:
            raise exception
        # Make sure the expected exception was raised:
        assert test.exception in str(exception)

    # Clear the tests outputs:
    shutil.rmtree(temp_directory)
    test_directory.cleanup()


@pytest.mark.parametrize(
    "tester, test",
    _get_tests_tuples(test_type=PackToUnpackTest),
)
def test_packager_pack_to_unpack(
    rundb_mock, tester: type[PackagerTester], test: PackToUnpackTest
):
    """
    Test a packager's packing and unpacking by running two MLRun functions one after the other, one will return the
    value the packager should pack and the other should get the data item to make the packager unpack.

    :param rundb_mock: A runDB mock fixture.
    :param tester: The `PackagerTester` class to get the functions to run from.
    :param test:   The `PackToUnpackTest` tuple with the test parameters.
    """
    # Set up the test, creating a project and a MLRun function:
    test_directory = tempfile.TemporaryDirectory()
    mlrun_function = _setup_test(
        tester=tester, test=test, test_directory=test_directory.name
    )

    # Run the packing handler:
    try:
        pack_run = mlrun_function.run(
            name="pack",
            handler=test.pack_handler,
            params=test.pack_parameters,
            returns=[test.log_hint],
            artifact_path=test_directory.name,
            local=True,
        )

        # Verify the outputs are logged (artifact type as "result" will stop the test here as it cannot be unpacked):
        key, artifact_type = _get_key_and_artifact_type(tester=tester, test=test)
        if artifact_type == ArtifactType.RESULT:
            assert key in pack_run.status.results
            return
        assert key in pack_run.outputs

        # Validate the packager manager notes and packager instructions:
        unpackaging_instructions = pack_run._artifact(key=key)["spec"][
            "unpackaging_instructions"
        ]
        assert (
            unpackaging_instructions["packager_name"]
            == tester.PACKAGER_IN_TEST.__class__.__name__
        )
        if tester.PACKAGER_IN_TEST.PACKABLE_OBJECT_TYPE is not ...:
            # Check the object name noted match the packager handled type (at least subclass of it):
            packable_object_type_name = PackagersManager._get_type_name(
                typ=tester.PACKAGER_IN_TEST.PACKABLE_OBJECT_TYPE
                if tester.PACKAGER_IN_TEST.PACKABLE_OBJECT_TYPE.__module__ != "typing"
                else typing.get_origin(tester.PACKAGER_IN_TEST.PACKABLE_OBJECT_TYPE)
            )
            assert unpackaging_instructions[
                "object_type"
            ] == packable_object_type_name or issubclass(
                PackagersManager._get_type_from_name(
                    type_name=unpackaging_instructions["object_type"]
                ),
                tester.PACKAGER_IN_TEST.PACKABLE_OBJECT_TYPE,
            )
        assert unpackaging_instructions["artifact_type"] == artifact_type
        assert unpackaging_instructions["instructions"] == test.expected_instructions

        # Run the unpacking handler:
        mlrun_function.run(
            name="unpack",
            handler=test.unpack_handler,
            inputs={"obj": pack_run.outputs[key]},
            params=test.unpack_parameters,
            artifact_path=test_directory.name,
            local=True,
        )
    except Exception as exception:
        # An error was raised, check if the test failed or should have failed:
        if test.exception is None:
            raise exception
        # Make sure the expected exception was raised:
        assert test.exception in str(exception)

    # Clear the tests outputs:
    test_directory.cleanup()
