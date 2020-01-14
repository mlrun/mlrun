# Copyright 2020 Iguazio
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

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock, Thread
from time import sleep

from croniter import croniter

from mlrun.runtimes import BaseRuntime
from mlrun.utils import logger


class Job:
    def __init__(self, schedule, runtime: BaseRuntime, args=None, kw=None):
        self.sched = croniter(schedule, ret_type=datetime)
        self.runtime = runtime
        self.args = () if args is None else args
        self.kw = {} if kw is None else kw
        self.next = self.sched.get_next()

    def advance(self):
        self.next = self.sched.get_next()

    def __repr__(self):
        cls = self.__class__.__name__
        sched = ' '.join(v[0] for v in self.sched.expanded)
        runtime, args, kw = self.runtime, self.args, self.kw
        return f'{cls}({sched!r}, {runtime!r}, {args!r}, {kw!r})'


class Scheduler(list):
    sleep_time_sec = 60  # a minute

    def __init__(self):
        self.lock = Lock()
        self.pool = ThreadPoolExecutor()
        Thread(target=self._loop, daemon=True).start()

    def add(self, schedule: str, runtime: BaseRuntime, args=None, kw=None):
        """Add a job to run according to schedule.

        args & kw are passed to runtime.run
        """
        args = () if args is None else args
        kw = {} if kw is None else kw
        job = Job(schedule, runtime, args, kw)
        with self.lock:
            self.append(job)
        return id(job)

    def _loop(self):
        while True:
            now = datetime.now()
            logger.info('scheduler loop at %s', now)
            with self.lock:
                for job in self:
                    if job.next <= now:
                        logger.info('scheduling job')
                        self.pool.submit(job.runtime.run, *job.args, **job.kw)
                        job.advance()
            sleep(self.sleep_time_sec)
