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
from abc import ABC
from typing import Any, Callable, List, NamedTuple, Tuple, Union

from mlrun import Packager


class PackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_pack` test of "test_packagers.py".

    :param pack_handler:                 The handler to run as a MLRun function for packing.
    :param parameters:                   The parameters to pass to the pack handler.
    :param log_hint:                     The log hint to pass to the pack handler.
    :param validation_function:          Function to assert a success packing. Will run without MLRun.
    :param default_artifact_type_object: Optional field to hold a dummy object to test the default artifact type method
                                         of the packager. Make sure to not pass an artifact type in the log hint, so it
                                         will be tested.
    """

    pack_handler: str
    parameters: dict
    log_hint: Union[str, dict]
    validation_function: Callable[[Any], bool]
    default_artifact_type_object: Any = None


class UnpackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_unpack` test of "test_packagers.py".

    :param prepare_input_function: Function to prepare the input to pass to the unpack handler. It should return a tuple
                                   of strings: the root directory to delete after the test where all files that were
                                   generated are stored, and the input path to pass as input to the function.
    :param unpack_handler:         The handler to run as a MLRun function for unpacking. Must accept "obj" as the
                                   argument to unpack.
    """

    prepare_input_function: Callable[[], Tuple[str, str]]
    unpack_handler: str


class PackToUnpackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_pack_to_unpack` test of "test_packagers.py".

    :param pack_handler:                 The handler to run as a MLRun function for packing.
    :param parameters:                   The parameters to pass to the pack handler.
    :param log_hint:                     The log hint to pass to the pack handler.
    :param expected_instructions:        The expected instructions the packed artifact should have.
    :param unpack_handler:               The handler to run as a MLRun function for unpacking. Must accept "obj" as the
                                         argument to unpack.
    :param default_artifact_type_object: Optional field to hold a dummy object to test the default artifact type method
                                         of the packager. Make sure to not pass an artifact type in the log hint, so it
                                         will be tested.
    """

    pack_handler: str
    parameters: dict
    log_hint: Union[str, dict]
    expected_instructions: dict
    unpack_handler: str
    default_artifact_type_object: Any = None


class PackagerTester(ABC):
    """
    A simple class for all testers to inherit from, so they will be able to be added to the tests in
    "test_packagers.py".
    """

    # The packager being tested by this tester:
    PACKAGER_IN_TEST: Packager = None

    # The list of tests tuples to include from this tester in the tests of "test_packagers.py":
    TESTS: List[Union[PackTest, UnpackTest, PackToUnpackTest]] = []
