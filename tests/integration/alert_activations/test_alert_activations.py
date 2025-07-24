# Copyright 2025 Iguazio
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

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mlrun.common.db.dialects import Dialects
from mlrun.common.schemas.partition import PartitionInterval
from server.py.framework.db.sqldb.models import AlertActivation, Base
from tests.common_fixtures import freeze_datetime

# Determine which backend is under test
TEST_DB = os.getenv("MLRUN_TEST_DB", Dialects.MYSQL)

MYSQL_ONLY = pytest.mark.skipif(
    not Dialects.MYSQL.startswith(TEST_DB),
    reason="MySQL-only test",
)
PG_ONLY = pytest.mark.skipif(
    not Dialects.POSTGRESQL.startswith(TEST_DB),
    reason="Postgres-only test",
)


@pytest.mark.integration
@MYSQL_ONLY
@pytest.mark.parametrize(
    "interval_name,id_val",
    [
        ("DAY", 1),
        ("MONTH", 2),
        ("YEARWEEK", 3),
    ],
)
@freeze_datetime(datetime(2025, 1, 6))
def test_mysql_insert_populates_partition_key(
    db_engine: Engine,
    interval_name: str,
    id_val: int,
) -> None:
    os.environ["PARTITION_INTERVAL"] = interval_name
    Base.metadata.drop_all(db_engine)
    Base.metadata.create_all(db_engine)

    ts: datetime = datetime.now(UTC)
    activation: AlertActivation = AlertActivation(
        id=id_val,
        activation_time=ts,
        project="project_x",
        name="alert_y",
        entity_id="entity_1",
        entity_kind="kind_1",
        event_kind="event_1",
        severity="high",
        number_of_events=1,
    )

    with Session(db_engine) as session:
        session.add(activation)
        session.flush()  # fires before_insert listener
        expected_key: int = int(
            PartitionInterval(interval_name).get_partition_info(ts)[0][1]
        )
        assert activation.partition_key == expected_key


@pytest.mark.integration
@PG_ONLY
@pytest.mark.parametrize(
    "interval_name,id_val",
    [
        ("DAY", 10),
        ("MONTH", 20),
        ("YEARWEEK", 30),
    ],
)
@freeze_datetime(datetime(2025, 1, 6))
def test_postgres_insert_populates_partition_key(
    db_engine: Engine,
    interval_name: str,
    id_val: int,
) -> None:
    os.environ["PARTITION_INTERVAL"] = interval_name
    Base.metadata.drop_all(db_engine)
    Base.metadata.create_all(db_engine)

    ts: datetime = datetime.now(UTC)
    activation: AlertActivation = AlertActivation(
        id=id_val,
        activation_time=ts,
        project="project_a",
        name="alert_b",
        entity_id="entity_2",
        entity_kind="kind_2",
        event_kind="event_2",
        severity="low",
        number_of_events=2,
    )

    with Session(db_engine) as session:
        session.add(activation)
        session.flush()  # fires before_insert listener
        expected_key = int(
            PartitionInterval(interval_name).get_partition_info(ts)[0][1]
        )
        assert activation.partition_key == expected_key
