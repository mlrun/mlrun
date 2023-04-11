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
from typing import List, Tuple, Type, Union

import pytest

import mlrun
from mlrun.package import ArtifactType, LogHintKey
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


def _setup_test(tester: Type[PackagerTester]) -> KubejobRuntime:
    # Create a project for this tester:
    project = mlrun.get_or_create_project(name=f"{tester.__name__.lower()}-project")

    # Create a MLRun function using the tester source file (all the functions must be located in it):
    return project.set_function(
        func=inspect.getfile(tester),
        name=f"{tester.__name__.lower()}-function",
        kind="job",
    )


@pytest.mark.parametrize(
    "test_tuple",
    _get_tests_tuples(test_type=PackTest),
)
def test_packager_pack(rundb_mock, test_tuple: Tuple[Type[PackagerTester], PackTest]):
    """
    Test a packager's packing.

    :param rundb_mock: A runDB mock fixture.
    :param test_tuple: A tuple of the `PackagerTester` class to get the functions to run from and the `PackTest`
                       tuple with the test parameters.
    """
    # Unpack the test tuple:
    tester, test = test_tuple

    # Set up the test, creating a project and a MLRun function:
    mlrun_function = _setup_test(tester=tester)

    # Run the packing handler:
    pack_run = mlrun_function.run(
        name="pack",
        handler=test.pack_handler,
        params=test.parameters,
        returns=[test.log_hint],
        local=True,
    )

    # Verify the packaged output:
    if test.log_hint[LogHintKey.ARTIFACT_TYPE] == ArtifactType.RESULT:
        assert test.log_hint[LogHintKey.KEY] in pack_run.status.results
        assert test.validation_function(
            pack_run.status.results[test.log_hint[LogHintKey.KEY]]
        )
    else:
        assert test.log_hint[LogHintKey.KEY] in pack_run.outputs
        assert test.validation_function(pack_run.outputs[test.log_hint[LogHintKey.KEY]])


@pytest.mark.parametrize(
    "test_tuple",
    _get_tests_tuples(test_type=UnpackTest),
)
def test_packager_unpack(
    rundb_mock, test_tuple: Tuple[Type[PackagerTester], UnpackTest]
):
    """
    Test a packager's unpacking.

    :param rundb_mock: A runDB mock fixture.
    :param test_tuple: A tuple of the `PackagerTester` class to get the functions to run from and the `UnpackTest`
                       tuple with the test parameters.
    """
    # Unpack the test tuple:
    tester, test = test_tuple

    # Create the input path to send for unpacking:
    input_path = test.prepare_input_function()

    # Set up the test, creating a project and a MLRun function:
    mlrun_function = _setup_test(tester=tester)

    # Run the packing handler:
    mlrun_function.run(
        name="unpack",
        handler=test.unpack_handler,
        inputs={"obj": input_path},
        local=True,
    )


@pytest.mark.parametrize(
    "test_tuple",
    _get_tests_tuples(test_type=PackToUnpackTest),
)
def test_packager_pack_to_unpack(
    rundb_mock, test_tuple: Tuple[Type[PackagerTester], PackToUnpackTest]
):
    """
    Test a packager's packing and unpacking by running two MLRun functions one after the other, one will return the
    value the packager should pack and the other should get the data item to make the packager unpack.

    :param rundb_mock: A runDB mock fixture.
    :param test_tuple: A tuple of the `PackagerTester` class to get the functions to run from and the `PackToUnpackTest`
                       tuple with the test parameters.
    """
    # Unpack the test tuple:
    tester, test = test_tuple

    # Set up the test, creating a project and a MLRun function:
    mlrun_function = _setup_test(tester=tester)

    # Run the packing handler:
    pack_run = mlrun_function.run(
        name="pack",
        handler=test.pack_handler,
        params=test.parameters,
        returns=[test.log_hint],
        local=True,
    )

    # Verify the outputs are logged (artifact type as "result" will stop the test here as it cannot be unpacked):
    if test.log_hint[LogHintKey.ARTIFACT_TYPE] == ArtifactType.RESULT:
        assert test.log_hint[LogHintKey.KEY] in pack_run.status.results
        return
    assert test.log_hint[LogHintKey.KEY] in pack_run.outputs

    # Validate the packager instructions noted:
    instructions = pack_run.outputs[
        test.log_hint[LogHintKey.KEY]
    ].spec.packaging_instructions
    assert instructions == test.expected_instructions

    # Run the unpacking handler:
    mlrun_function.run(
        name="unpack",
        handler=test.unpack_handler,
        inputs={"obj": pack_run.outputs[test.log_hint[LogHintKey.KEY]]},
        local=True,
    )
