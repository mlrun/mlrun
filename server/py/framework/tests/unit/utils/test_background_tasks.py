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
import datetime

import pytest

import mlrun.common.schemas
import mlrun.utils

import framework.utils.background_tasks.common


@pytest.mark.parametrize(
    "start_time",
    [
        # SQLite returns tz-naive datetimes for DateTime columns
        mlrun.utils.now_date().replace(tzinfo=None) - datetime.timedelta(hours=1),
        # PostgreSQL/MySQL return tz-aware datetimes for the same logical timestamp
        mlrun.utils.now_date() - datetime.timedelta(hours=1),
    ],
)
def test_background_task_exceeded_timeout_across_dialects(start_time):
    # regardless of whether start_time is tz-naive (SQLite) or tz-aware (PostgreSQL/MySQL), the
    # comparison against `now_date()` must not raise and must reach the correct verdict
    assert framework.utils.background_tasks.common.background_task_exceeded_timeout(
        start_time,
        timeout=60,
        task_state=mlrun.common.schemas.BackgroundTaskState.running,
    )
    assert not framework.utils.background_tasks.common.background_task_exceeded_timeout(
        start_time,
        timeout=3600 * 24,
        task_state=mlrun.common.schemas.BackgroundTaskState.running,
    )


@pytest.mark.parametrize(
    "start_time",
    [
        mlrun.utils.now_date().replace(tzinfo=None) - datetime.timedelta(hours=1),
        mlrun.utils.now_date() - datetime.timedelta(hours=1),
    ],
)
def test_background_task_exceeded_timeout_terminal_state_short_circuits(start_time):
    assert not framework.utils.background_tasks.common.background_task_exceeded_timeout(
        start_time,
        timeout=60,
        task_state=mlrun.common.schemas.BackgroundTaskState.succeeded,
    )
