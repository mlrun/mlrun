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

import inspect
import shutil
import tempfile
import typing

import pytest

import mlrun
from mlrun.package import ArtifactType, PackagersManager
from mlrun.package.log_hint import LogHint
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

# All the testers to be included in the tests:
_PACKAGERS_TESTERS = [
    DefaultPackagerTester,
    NonePackagerTester,
    BoolPackagerTester,
    BytearrayPackagerTester,
    BytesPackagerTester,
    DictPackagerTester,
    FloatPackagerTester,
    FrozensetPackagerTester,
    IntPackagerTester,
    ListPackagerTester,
    SetPackagerTester,
    StrPackagerTester,
    TuplePackagerTester,
    PathPackagerTester,
    NumPyNDArrayPackagerTester,
    NumPyNumberPackagerTester,
    NumPyNDArrayDictPackagerTester,
    NumPyNDArrayListPackagerTester,
    PandasDataFramePackagerTester,
    PandasSeriesPackagerTester,
]


def _get_tests_tuples(
    test_type: type[PackTest] | type[UnpackTest] | type[PackToUnpackTest],
) -> list[tuple[type[PackagerTester], PackTest]]:
    return [
        (tester, test)
        for tester in _PACKAGERS_TESTERS
        for test in tester.TESTS
        if isinstance(test, test_type)
    ]


def _setup_test(
    tester: type[PackagerTester],
    test: PackTest | UnpackTest | PackToUnpackTest,
    test_directory: str,
) -> KubejobRuntime:
    # Enabled logging tuples only if the tuple test is about to be setup:
    if isinstance(test, PackTest | PackToUnpackTest) and tester is TuplePackagerTester:
        mlrun.mlconf.packagers.pack_tuples = True

    # Create a project for this tester:
    project = mlrun.get_or_create_project(
        name="default", context=test_directory, allow_cross_project=True
    )

    # Create a MLRun function using the tester source file (all the functions must be located in it):
    return project.set_function(
        func=inspect.getfile(tester),
        name=tester.__name__.lower(),
        kind="job",
        image="mlrun/mlrun",
    )


def _get_log_hint(
    tester: type[PackagerTester], test: PackTest | PackToUnpackTest
) -> LogHint:
    # Parse the log hint (in case it is a string):
    log_hint = LogHint.parse_obj(obj=test.log_hint)

    # Get the artifact type (either from the log hint or from the packager - the default artifact type):
    log_hint.artifact_type = (
        log_hint.artifact_type
        if log_hint.artifact_type
        else tester.PACKAGER_IN_TEST.get_default_packing_artifact_type(
            obj=test.default_artifact_type_object
        )
    )

    return log_hint


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
            output_path=test_directory.name,
            local=True,
        )

        # Verify the packaged output:
        log_hint = _get_log_hint(tester=tester, test=test)

        # If bundling was performed, check each element from the bundle accordingly (they will be sent to the validation
        # function as well):
        unbundled_artifacts = {}
        if log_hint.itemized:
            unbundled_artifacts = {
                k: pack_run.outputs[k]
                for k in pack_run.outputs
                if k.startswith(log_hint.key) and k != log_hint.key
            }
            assert unbundled_artifacts

        # Verify the output:
        assert log_hint.key in pack_run.outputs
        assert test.validation_function(
            pack_run.outputs[log_hint.key], **test.validation_parameters
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
            output_path=test_directory.name,
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
            output_path=test_directory.name,
            local=True,
        )

        # Verify the outputs are logged (artifact type as "result" will stop the test here as it cannot be unpacked):
        log_hint = _get_log_hint(tester=tester, test=test)

        # Verify result:
        if log_hint.artifact_type == ArtifactType.RESULT:
            if log_hint.itemized:
                # For unbundling results, just verify results exist with the prefix
                unbundled_results = {
                    k: v
                    for k, v in pack_run.status.results.items()
                    if k.startswith(log_hint.key)
                }
                assert len(unbundled_results) > 0
            else:
                assert log_hint.key in pack_run.status.results
            return

        # Verify artifact (Notice: for bundles we do not check the instructions as the packager in test did only the
        # bundling and unbundling, not the packing - so there are no instructions):
        if log_hint.itemized:
            # For unbundling artifacts, collect all outputs that start with the unbundle prefix:
            unbundled_outputs = {
                k: pack_run.outputs[k]
                for k in pack_run.outputs
                if k.startswith(log_hint.key)
            }
            assert (
                len(unbundled_outputs) > 2
            )  # The bundle result + at least one artifact from the bundle.
            # Run unpack handler with bundled input
            mlrun_function.run(
                name="unpack",
                handler=test.unpack_handler,
                inputs={"obj": pack_run.outputs[log_hint.key]},
                params=test.unpack_parameters,
                output_path=test_directory.name,
                local=True,
            )
        else:
            # Regular single artifact unpacking:
            assert log_hint.key in pack_run.outputs
            # Validate the packager manager notes and packager instructions:
            unpackaging_instructions = pack_run._artifact(key=log_hint.key)["spec"][
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
                    if tester.PACKAGER_IN_TEST.PACKABLE_OBJECT_TYPE.__module__
                    != "typing"
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
            assert unpackaging_instructions["artifact_type"] == log_hint.artifact_type
            assert (
                unpackaging_instructions["instructions"] == test.expected_instructions
            )
            # Run the unpacking handler:
            mlrun_function.run(
                name="unpack",
                handler=test.unpack_handler,
                inputs={"obj": pack_run.outputs[log_hint.key]},
                params=test.unpack_parameters,
                output_path=test_directory.name,
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
