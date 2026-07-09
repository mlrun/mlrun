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

_ONE_HOUR_AGO = mlrun.utils.now_date() - datetime.timedelta(hours=1)
# start_time as SQLite (tz-naive) vs PostgreSQL/MySQL (tz-aware) would return it for the same
# logical timestamp - the comparison against `now_date()` must behave the same either way
_START_TIME_DIALECT_VARIANTS = [_ONE_HOUR_AGO.replace(tzinfo=None), _ONE_HOUR_AGO]


@pytest.mark.parametrize("start_time", _START_TIME_DIALECT_VARIANTS)
def test_background_task_exceeded_timeout_across_dialects(start_time):
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


@pytest.mark.parametrize("start_time", _START_TIME_DIALECT_VARIANTS)
def test_background_task_exceeded_timeout_terminal_state_short_circuits(start_time):
    assert not framework.utils.background_tasks.common.background_task_exceeded_timeout(
        start_time,
        timeout=60,
        task_state=mlrun.common.schemas.BackgroundTaskState.succeeded,
    )
