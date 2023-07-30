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

import typing

import pytest

from mlrun.api.crud.model_monitoring.helpers import (
    Minutes,
    Seconds,
    _add_minutes_offset,
    seconds2minutes,
)


@pytest.mark.parametrize(
    ("seconds", "expected_minutes"),
    [(0, 0), (1, 1), (60, 1), (21, 1), (365, 7)],
)
def test_minutes2seconds(seconds: Seconds, expected_minutes: Minutes) -> None:
    assert seconds2minutes(seconds) == expected_minutes


@pytest.mark.parametrize(
    ("minute", "offset", "expected_result"),
    [
        (0, 0, 0),
        ("0", 0, 0),
        ("*", 22, "*"),
        ("*/4", 2, "*/4"),
        ("0", 20, 20),
        ("40", 30, 10),
    ],
)
def test_add_minutes_offset(
    minute: typing.Optional[typing.Union[int, str]],
    offset: Minutes,
    expected_result: typing.Optional[typing.Union[int, str]],
) -> None:
    assert _add_minutes_offset(minute, offset) == expected_result
