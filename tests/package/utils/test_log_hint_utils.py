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
from typing import Union

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
def test_parse_log_hint(
    log_hint: Union[str, dict], expected_log_hint: Union[str, dict]
):
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
