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
from typing import Any, Callable, List, NamedTuple, Union


class PackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_pack` test of "test_packagers.py".

    :param pack_handler:        The handler to run as a MLRun function for packing.
    :param parameters:          The parameters to pass to the pack handler.
    :param log_hint:            The log hint to pass to the pack handler.
    :param validation_function: Function to assert a success packing. Will run without MLRun.
    """

    pack_handler: str
    parameters: dict
    log_hint: Union[str, dict]
    validation_function: Callable[[Any], bool]


class UnpackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_unpack` test of "test_packagers.py".

    :param prepare_input_function: Function to prepare the input to pass to the unpack handler.
    :param unpack_handler:         The handler to run as a MLRun function for unpacking. Must accept "obj" as the
                                   argument to unpack.
    """

    prepare_input_function: Callable[[], str]
    unpack_handler: str


class PackToUnpackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_pack_to_unpack` test of "test_packagers.py".

    :param pack_handler:          The handler to run as a MLRun function for packing.
    :param parameters:            The parameters to pass to the pack handler.
    :param log_hint:              The log hint to pass to the pack handler.
    :param expected_instructions: The expected instructions the packed artifact should have.
    :param unpack_handler:        The handler to run as a MLRun function for unpacking. Must accept "obj" as the
                                  argument to unpack.
    """

    pack_handler: str
    parameters: dict
    log_hint: Union[str, dict]
    expected_instructions: dict
    unpack_handler: str


class PackagerTester(ABC):
    """
    A simple class for all testers to inherit from, so they will be able to be added to the tests in
    "test_packagers.py".
    """

    # The list of tests tuples to include from this tester in the tests of "test_packagers.py":
    TESTS: List[Union[PackTest, UnpackTest, PackToUnpackTest]] = []
