# Copyright 2026 Iguazio
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

import mlrun.common.schemas.schedule


def test_to_crontab_from_crontab_roundtrip():
    """Verify that from_crontab -> to_crontab produces the original expression."""
    expr = "*/5 2 1 3 0"
    trigger = mlrun.common.schemas.schedule.ScheduleCronTrigger.from_crontab(expr)
    assert trigger.to_crontab() == expr


def test_to_crontab_none_fields_no_literal_none():
    """Fields left as None must not produce the literal string 'None'."""
    trigger = mlrun.common.schemas.schedule.ScheduleCronTrigger(
        minute="0",
        hour="9",
        day=None,
        month=None,
        day_of_week=None,
    )
    result = trigger.to_crontab()
    assert "None" not in result, (
        f"to_crontab() produced invalid crontab with literal 'None': {result}"
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"minute": "0", "hour": "9", "day": "15", "month": "6", "day_of_week": "1"},
            "0 9 15 6 1",
        ),
        (
            {
                "minute": "0",
                "hour": "9",
                "day": None,
                "month": None,
                "day_of_week": None,
            },
            "0 9 * * *",
        ),
        (
            {
                "minute": None,
                "hour": None,
                "day": None,
                "month": None,
                "day_of_week": None,
            },
            "* * * * *",
        ),
        (
            {"minute": 30, "hour": 14, "day": None, "month": None, "day_of_week": None},
            "30 14 * * *",
        ),
    ],
    ids=["all_fields", "partial_none", "all_none", "integer_fields"],
)
def test_to_crontab(kwargs, expected):
    """Verify to_crontab produces the correct crontab expression."""
    trigger = mlrun.common.schemas.schedule.ScheduleCronTrigger(**kwargs)
    assert trigger.to_crontab() == expected
