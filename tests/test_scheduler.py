from datetime import datetime

import pytest
from croniter import CroniterBadCronError

from mlrun import scheduler


class Runtime(list):
    def run(self, *args, **kw):
        self.append((datetime.now(), args, kw))


def test_scheduler():
    sched = scheduler.Scheduler()
    rt = Runtime()
    sched.add('* * * * *', rt, (1, 2), {'a': 1, 'b': 2})
    assert 1 == len(sched), 'bad jobs'
    # TODO: Speed up clock so we can see the job scheduled


def test_bad_schedule():
    sched = scheduler.Scheduler()
    with pytest.raises(CroniterBadCronError):
        sched.add('* * * *', None)
