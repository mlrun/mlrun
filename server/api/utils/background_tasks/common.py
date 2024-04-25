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

import datetime

import mlrun.common.schemas


def background_task_exceeded_timeout(start_time, timeout, task_state) -> bool:
    # We don't verify if timeout_mode is enabled because if timeout is defined and
    # mlrun.mlconf.background_tasks.timeout_mode == "disabled",
    # it signifies that the background task was initiated while timeout mode was enabled,
    # and we intend to verify it as if timeout mode was enabled
    if (
        timeout
        and task_state not in mlrun.common.schemas.BackgroundTaskState.terminal_states()
        and datetime.datetime.utcnow()
        > datetime.timedelta(seconds=int(timeout)) + start_time
    ):
        return True
    return False
