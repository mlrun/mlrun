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
import sys
from abc import ABC
from typing import Any, Callable, NamedTuple, Union

import cloudpickle

from mlrun import Packager

# When using artifact type "object", these instructions will be common to most artifacts in the tests:
COMMON_OBJECT_INSTRUCTIONS = {
    "pickle_module_name": "cloudpickle",
    "pickle_module_version": cloudpickle.__version__,
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
}


class PackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_pack` test of "test_packagers.py".

    :param pack_handler:                 The handler to run as a MLRun function for packing.
    :param log_hint:                     The log hint to pass to the pack handler.
    :param validation_function:          Function to assert a success packing. Will run without MLRun. It expects to
                                         receive the logged result / Artifact object.
    :param pack_parameters:              The parameters to pass to the pack handler.
    :param validation_parameters:        Additional parameters to pass to the validation function.
    :param default_artifact_type_object: Optional field to hold a dummy object to test the default artifact type method
                                         of the packager. Make sure to not pass an artifact type in the log hint, so it
                                         will be tested.
    :param exception:                    If an exception should be raised during the test, this should be part of the
                                         expected exception message. Default is None (the test should succeed).
    """

    pack_handler: str
    log_hint: Union[str, dict]
    validation_function: Callable[..., bool]
    pack_parameters: dict = {}
    validation_parameters: dict = {}
    default_artifact_type_object: Any = None
    exception: str = None


class UnpackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_unpack` test of "test_packagers.py".

    :param prepare_input_function: Function to prepare the input to pass to the unpack handler. It should return a tuple
                                   of strings: the input path to pass as input to the function and the root directory to
                                   delete after the test where all files that were generated are stored.
    :param unpack_handler:         The handler to run as a MLRun function for unpacking. Must accept "obj" as the
                                   argument to unpack.
    :param prepare_parameters:     The parameters to pass to the prepare function.
    :param unpack_parameters:      The parameters to pass to the unpack handler.
    :param exception:              If an exception should be raised during the test, this should be part of the expected
                                   exception message. Default is None (the test should succeed).
    """

    prepare_input_function: Callable[..., tuple[str, str]]
    unpack_handler: str
    prepare_parameters: dict = {}
    unpack_parameters: dict = {}
    exception: str = None


class PackToUnpackTest(NamedTuple):
    """
    Tuple for creating a test to run in the `test_packager_pack_to_unpack` test of "test_packagers.py".

    :param pack_handler:                 The handler to run as a MLRun function for packing.
    :param log_hint:                     The log hint to pass to the pack handler. Result will skip the
                                         `expected_instructions` and `unpack_handler` variables (hence they are
                                         optional).
    :param pack_parameters:              The parameters to pass to the pack handler.
    :param expected_instructions:        The expected instructions the packed artifact should have.
    :param unpack_handler:               The handler to run as a MLRun function for unpacking. Must accept "obj" as the
                                         argument to unpack.
    :param unpack_parameters:            The parameters to pass to the unpack handler.
    :param default_artifact_type_object: Optional field to hold a dummy object to test the default artifact type method
                                         of the packager. Make sure to not pass an artifact type in the log hint, so it
                                         will be tested.
    :param exception:                    If an exception should be raised during the test, this should be part of the
                                         expected exception message. Default is None (the test should succeed).
    """

    pack_handler: str
    log_hint: Union[str, dict]
    pack_parameters: dict = {}
    expected_instructions: dict = {}
    unpack_handler: str = None
    unpack_parameters: dict = {}
    default_artifact_type_object: Any = None
    exception: str = None


class PackagerTester(ABC):
    """
    A simple class for all testers to inherit from, so they will be able to be added to the tests in
    "test_packagers.py".
    """

    # The packager being tested by this tester:
    PACKAGER_IN_TEST: Packager = None

    # The list of tests tuples to include from this tester in the tests of "test_packagers.py":
    TESTS: list[Union[PackTest, UnpackTest, PackToUnpackTest]] = []


class NewClass:
    """
    Class to use for testing the default class.
    """

    # It is declared in this file so that it won't be part of the MLRun function module when a tester of
    # `default_packager_tester.py` is running. For more information, see the long exception at `packagers_manager.py`'s
    # `PackagersManager._unpack_package` function.

    def __init__(self, a: int, b: int, c: int):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c

    def __str__(self):
        return str(self.a + self.b + self.c)
