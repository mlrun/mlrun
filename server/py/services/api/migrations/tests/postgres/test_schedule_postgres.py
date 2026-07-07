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
import sqlalchemy.orm

import mlrun.common.schemas

import framework.db.sqldb.db

pytest.importorskip(
    "psycopg2",
    reason="psycopg2 not installed",
)


@pytest.mark.integration
@pytest.mark.usefixtures("pmr_postgres_container")
def test_schedule_cron_trigger_round_trip_on_postgres(
    postgres_db_session: sqlalchemy.orm.session.Session,
):
    # Regression for ML-12860 - see the cron_trigger setter in
    # server/py/framework/db/sqldb/models.py for why this only ever reproduced on PostgreSQL.
    db = framework.db.sqldb.db.SQLDB()
    project = "postgres-cron-trigger-test"
    name = "main"
    cron_trigger = mlrun.common.schemas.ScheduleCronTrigger.from_crontab("*/15 * * * *")

    db.store_schedule(
        postgres_db_session,
        project=project,
        name=name,
        kind=mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object={"task": {}},
        cron_trigger=cron_trigger,
    )

    schedule = db.get_schedule(postgres_db_session, project, name)

    assert schedule.cron_trigger.minute == "*/15"

    # store_schedule's own pre-existence check (get_schedule with raise_on_not_found=False) is
    # exactly where ML-12860 crashed on a second scheduling attempt for the same project/name -
    # exercise it explicitly here.
    db.store_schedule(
        postgres_db_session,
        project=project,
        name=name,
        kind=mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object={"task": {}},
        cron_trigger=cron_trigger,
    )
