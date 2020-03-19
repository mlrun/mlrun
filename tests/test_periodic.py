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

from datetime import datetime
from time import sleep

import pytest

from mlrun.httpd import periodic


class Task:
    def __init__(self, sleep_time, fail):
        self.runs = []
        self.sleep_time = sleep_time
        self.fail = fail

    def run(self):
        self.runs.append(datetime.now())
        sleep(self.sleep_time)
        if self.fail:
            raise ValueError('oopsy')


@pytest.mark.parametrize('fail', [False, True])
def test_periodic(fail):
    t = Task(0.1, fail)
    freq = 0.2
    periodic.schedule(t, freq)
    sleep(freq * 4)
    assert len(t.runs) >= 3, 'no runs'
