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
from mlrun.package.log_hint import LogHint


@pytest.mark.parametrize(
    "log_hint, expected_log_hint",
    [
        # No unbundling
        ("some_key", LogHint(key="some_key")),
        (
            "some_key:artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key :artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key: artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key : artifact",
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        (
            "some_key:",
            "Incorrect log hint pattern. The ':' in a log hint should specify",
        ),
        (
            "some_key : artifact : error",
            "Incorrect log hint pattern. Log hints can have only a single ':' in them",
        ),
        (LogHint(key="some_key"), LogHint(key="some_key")),
        (
            LogHint(key="some_key", artifact_type="artifact"),
            LogHint(key="some_key", artifact_type="artifact"),
        ),
        # Full unbundling (no level specified)
        ("*results", LogHint(key="results", itemized=True)),
        ("* results", LogHint(key="results", itemized=True)),
        (" *results", LogHint(key="results", itemized=True)),
        (" * results", LogHint(key="results", itemized=True)),
        # Level-specific unbundling
        ("1 * results", LogHint(key="results", itemized=1)),
        ("2 *nested", LogHint(key="nested", itemized=2)),
        ("3* deep", LogHint(key="deep", itemized=3)),
        ("10*multi", LogHint(key="multi", itemized=10)),
        # Error case - invalid level
        ("abc*results", "Invalid unbundle level"),
        ("1.5*results", "Invalid unbundle level"),
        # Error case - empty key after asterisk
        ("*", "Key is missing after the unbundle operator '*'"),
        ("  * ", "Key is missing after the unbundle operator '*'"),
        ("1*", "Key is missing after the unbundle operator '*'"),
    ],
)
def test_model_validate_from_string(
    log_hint: str | dict, expected_log_hint: str | dict
):
    """
    Test the `LogHint.model_validate` class method for handling strings.

    :param log_hint:          The log hint to parse.
    :param expected_log_hint: The expected parsed log hint. A string value indicates the parsing should fail with the
                              provided error message in the variable.
    """
    try:
        parsed_log_hint = LogHint.model_validate(obj=log_hint)
        assert parsed_log_hint == expected_log_hint
    except MLRunInvalidArgumentError as error:
        if isinstance(expected_log_hint, str):
            assert expected_log_hint in str(error)
        else:
            raise error
