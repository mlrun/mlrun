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
import inspect
import shutil
import tempfile
from typing import List, Tuple, Type, Union

import pytest

import mlrun
from mlrun.package import ArtifactType, LogHintKey
from mlrun.package.utils import LogHintUtils
from mlrun.runtimes import KubejobRuntime

from .packager_tester import PackagerTester, PackTest, PackToUnpackTest, UnpackTest
from .packagers_testers.default_packager_tester import DefaultPackagerTester

# The list of testers to include in the tests:
PACKAGERS_TESTERS = [DefaultPackagerTester]


def _get_tests_tuples(
    test_type: Union[Type[PackTest], Type[UnpackTest], Type[PackToUnpackTest]]
) -> List[Tuple[Type[PackagerTester], PackTest]]:
    return [
        (tester, test)
        for tester in PACKAGERS_TESTERS
        for test in tester.TESTS
        if isinstance(test, test_type)
    ]


def _setup_test(tester: Type[PackagerTester], test_directory: str) -> KubejobRuntime:
    # Create a project for this tester:
    project = mlrun.get_or_create_project(name="default", context=test_directory)

    # Create a MLRun function using the tester source file (all the functions must be located in it):
    return project.set_function(
        func=inspect.getfile(tester),
        name=tester.__name__.lower(),
        kind="job",
        image="mlrun/mlrun",
    )


def _get_key_and_artifact_type(
    tester: Type[PackagerTester], test: Union[PackTest, PackToUnpackTest]
) -> Tuple[str, str]:
    # Parse the log hint (in case it is a string):
    log_hint = LogHintUtils.parse_log_hint(log_hint=test.log_hint)

    # Extract the key:
    key = log_hint[LogHintKey.KEY]

    # Get the artifact type (either from the log hint of from the packager - the default artifact type):
    artifact_type = log_hint.get(
        LogHintKey.ARTIFACT_TYPE,
        tester.PACKAGER_IN_TEST.get_default_artifact_type(
            obj=test.default_artifact_type_object
        ),
    )

    return key, artifact_type


@pytest.mark.parametrize(
    "tester, test",
    _get_tests_tuples(test_type=PackTest),
)
def test_packager_pack(rundb_mock, tester: Type[PackagerTester], test: PackTest):
    """
    Test a packager's packing.

    :param rundb_mock: A runDB mock fixture.
    :param tester: The `PackagerTester` class to get the functions to run from.
    :param test:   The `PackTest` tuple with the test parameters.
    """
    # Set up the test, creating a project and a MLRun function:
    test_directory = tempfile.TemporaryDirectory()
    mlrun_function = _setup_test(tester=tester, test_directory=test_directory.name)

    # Run the packing handler:
    pack_run = mlrun_function.run(
        name="pack",
        handler=test.pack_handler,
        params=test.parameters,
        returns=[test.log_hint],
        artifact_path=test_directory.name,
        local=True,
    )

    # Verify the packaged output:
    key, artifact_type = _get_key_and_artifact_type(tester=tester, test=test)
    if artifact_type == ArtifactType.RESULT:
        assert key in pack_run.status.results
        assert test.validation_function(pack_run.status.results[key])
    else:
        assert key in pack_run.outputs
        assert test.validation_function(pack_run._artifact(key=key))

    # Clear the tests outputs:
    test_directory.cleanup()


@pytest.mark.parametrize(
    "tester, test",
    _get_tests_tuples(test_type=UnpackTest),
)
def test_packager_unpack(rundb_mock, tester: Type[PackagerTester], test: UnpackTest):
    """
    Test a packager's unpacking.

    :param rundb_mock: A runDB mock fixture.
    :param tester: The `PackagerTester` class to get the functions to run from.
    :param test:   The `UnpackTest` tuple with the test parameters.
    """
    # Create the input path to send for unpacking:
    temp_directory, input_path = test.prepare_input_function()

    # Set up the test, creating a project and a MLRun function:
    test_directory = tempfile.TemporaryDirectory()
    mlrun_function = _setup_test(tester=tester, test_directory=test_directory.name)

    # Run the packing handler:
    mlrun_function.run(
        name="unpack",
        handler=test.unpack_handler,
        inputs={"obj": input_path},
        artifact_path=test_directory.name,
        local=True,
    )

    # Clear the tests outputs:
    shutil.rmtree(temp_directory)
    test_directory.cleanup()


@pytest.mark.parametrize(
    "tester, test",
    _get_tests_tuples(test_type=PackToUnpackTest),
)
def test_packager_pack_to_unpack(
    rundb_mock, tester: Type[PackagerTester], test: PackToUnpackTest
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
    mlrun_function = _setup_test(tester=tester, test_directory=test_directory.name)

    # Run the packing handler:
    pack_run = mlrun_function.run(
        name="pack",
        handler=test.pack_handler,
        params=test.parameters,
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

    # Validate the packager instructions noted:
    instructions = pack_run._artifact(key=key)["spec"]["packaging_instructions"][
        "instructions"
    ]
    for (
        expected_instruction_key,
        expected_instruction_value,
    ) in test.expected_instructions.items():
        assert instructions[expected_instruction_key] == expected_instruction_value

    # Run the unpacking handler:
    mlrun_function.run(
        name="unpack",
        handler=test.unpack_handler,
        inputs={"obj": pack_run.outputs[key]},
        artifact_path=test_directory.name,
        local=True,
    )

    # Clear the tests outputs:
    test_directory.cleanup()
