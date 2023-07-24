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
