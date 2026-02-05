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

import pytest

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.package.utils.log_hint_utils import LogHintKey, LogHintUtils


@pytest.mark.parametrize(
    "log_hint, expected_log_hint",
    [
        ("some_key", {LogHintKey.KEY: "some_key"}),
        (
            "some_key:artifact",
            {LogHintKey.KEY: "some_key", LogHintKey.ARTIFACT_TYPE: "artifact"},
        ),
        (
            "some_key :artifact",
            {LogHintKey.KEY: "some_key", LogHintKey.ARTIFACT_TYPE: "artifact"},
        ),
        (
            "some_key: artifact",
            {LogHintKey.KEY: "some_key", LogHintKey.ARTIFACT_TYPE: "artifact"},
        ),
        (
            "some_key : artifact",
            {LogHintKey.KEY: "some_key", LogHintKey.ARTIFACT_TYPE: "artifact"},
        ),
        (
            "some_key:",
            "Incorrect log hint pattern. The ':' in a log hint should specify",
        ),
        (
            "some_key : artifact : error",
            "Incorrect log hint pattern. Log hints can have only a single ':' in them",
        ),
        ({LogHintKey.KEY: "some_key"}, {LogHintKey.KEY: "some_key"}),
        (
            {LogHintKey.KEY: "some_key", LogHintKey.ARTIFACT_TYPE: "artifact"},
            {LogHintKey.KEY: "some_key", LogHintKey.ARTIFACT_TYPE: "artifact"},
        ),
        (
            {LogHintKey.ARTIFACT_TYPE: "artifact"},
            "A log hint dictionary must include the 'key'",
        ),
    ],
)
def test_parse_log_hint(log_hint: str | dict, expected_log_hint: str | dict):
    """
    Test the `LogHintUtils.parse_log_hint` function with multiple types.

    :param log_hint:          The log hint to parse.
    :param expected_log_hint: The expected parsed log hint dictionary. A string value indicates the parsing should fail
                              with the provided error message in the variable.
    """
    try:
        parsed_log_hint = LogHintUtils.parse_log_hint(log_hint=log_hint)
        assert parsed_log_hint == expected_log_hint
    except MLRunInvalidArgumentError as error:
        if isinstance(expected_log_hint, str):
            assert expected_log_hint in str(error)
        else:
            raise error


@pytest.mark.parametrize(
    "log_hint, expected_key, expected_level",
    [
        # No unbundling
        ("results", "results", False),
        ("my_data", "my_data", False),
        ("some_key_with_underscore", "some_key_with_underscore", False),
        # Full unbundling (no level specified)
        ("*results", "results", True),
        ("*my_data", "my_data", True),
        # Level-specific unbundling
        ("1*results", "results", 1),
        ("2*nested", "nested", 2),
        ("3*deep", "deep", 3),
        ("10*multi", "multi", 10),
        # Whitespace handling
        ("1 *results", "results", 1),  # Space before * is stripped from level
        # Error case - invalid level
        ("abc*results", "Invalid unbundle level", None),
        ("1.5*results", "Invalid unbundle level", None),
        # TODO: Error case - multiple asterisks (will be validated by the LogHint class in next PR
        # ("1*2*results", "Invalid log hint key", None),
        # Error case - empty key after asterisk
        ("*", "Key is missing after the '*'", None),
        ("  * ", "Key is missing after the '*'", None),
        ("1*", "Key is missing after the '*'", None),
    ],
)
def test_extract_unbundling_from_key(
    log_hint: str, expected_key: str, expected_level: bool | int | None
):
    """
    Test the `LogHintUtils.extract_unbundling_from_key` function.

    :param log_hint:       The log hint key to extract unbundling information from.
    :param expected_key:   The expected artifact key after extraction. A string starting with "Invalid" indicates
                           an error should be raised.
    :param expected_level: The expected unbundle level (True for full unbundling, False for no unbundling,
                           or an integer for specific unbundle level). None indicates an error case.
    """
    try:
        key, level = LogHintUtils.extract_unbundling_from_key(log_hint=log_hint)
        assert key == expected_key
        assert level == expected_level
    except MLRunInvalidArgumentError as error:
        if expected_level is None:
            assert expected_key in str(error)
        else:
            raise error
