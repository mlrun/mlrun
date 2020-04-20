# Copyright 2019 Iguazio
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

from threading import Thread
from time import monotonic, sleep

from ..utils import logger


class Task:
    # @yaronha - Add initializtion here
    def __init__(self):
        pass

    def run(self):
        # @yaronha - Add code here
        pass


def _schedule(task: Task, delay_seconds):
    while True:
        start = monotonic()
        try:
            try:
                task.run()
            except Exception as err:
                logger.exception('task error - %s', err)
        except Exception:
            pass

        duration = monotonic() - start
        sleep_time = max(delay_seconds - duration, 0)
        sleep(sleep_time)


def schedule(task: Task, delay_seconds):
    """Run task.run every delay_seconds in a background thread"""
    thr = Thread(target=_schedule, args=(task, delay_seconds), daemon=True)
    thr.start()
    return thr
