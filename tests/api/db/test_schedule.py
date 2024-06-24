# Copyright 2024 Iguazio
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
from sqlalchemy.orm import Session

import mlrun.common.schemas
from server.api.db.base import DBInterface
from server.api.db.sqldb.models import Schedule


def test_delete_schedules(db: DBInterface, db_session: Session):
    names = ["some_name", "some_name2", "some_name3"]
    labels = {
        "key": "value",
    }
    for name in names:
        db.store_schedule(
            db_session,
            project="project1",
            name=name,
            labels=labels,
            kind=mlrun.common.schemas.ScheduleKinds.job,
            cron_trigger=mlrun.common.schemas.ScheduleCronTrigger(minute=10),
        )
        db.store_schedule(
            db_session,
            project="project2",
            name=name,
            labels=labels,
            kind=mlrun.common.schemas.ScheduleKinds.job,
            cron_trigger=mlrun.common.schemas.ScheduleCronTrigger(minute=10),
        )

    schedules = db.list_schedules(db_session, project="project1")
    assert len(schedules) == len(names)
    schedules = db.list_schedules(db_session, project="project2")
    assert len(schedules) == len(names)

    assert db_session.query(Schedule.Label).count() != 0
    assert db_session.query(Schedule).count() != 0

    db.delete_schedules(db_session, "*", names=names[:2])
    schedules = db.list_schedules(db_session, project="project1")
    assert len(schedules) == 1
    schedules = db.list_schedules(db_session, project="project2")
    assert len(schedules) == 1

    assert db_session.query(Schedule.Label).count() == 2
    assert db_session.query(Schedule).count() == 2

    db.store_schedule(
        db_session,
        project="project1",
        name="no_delete",
        labels=labels,
        kind=mlrun.common.schemas.ScheduleKinds.job,
        cron_trigger=mlrun.common.schemas.ScheduleCronTrigger(minute=10),
    )
    db.delete_schedules(db_session, "*", names=names[:2])

    assert db_session.query(Schedule.Label).count() == 3
    assert db_session.query(Schedule).count() == 3
