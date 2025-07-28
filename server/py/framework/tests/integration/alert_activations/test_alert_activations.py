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

try:
    from datetime import UTC, datetime  # UTC is only defined in Python 3.11+
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc

import pytest
import sqlalchemy.engine
import sqlalchemy.orm

import mlrun.common.db.dialects
import mlrun.common.schemas.partition_interval
import server.py.framework.db.sqldb.models
import tests.common_fixtures


@pytest.mark.integration
@pytest.mark.parametrize(
    "interval_name,id_val",
    [
        ("DAY", 1),
        ("MONTH", 2),
        ("YEARWEEK", 3),
    ],
)
@tests.common_fixtures.freeze_datetime(datetime(2025, 1, 6))
def test_insert_populates_partition_key(
    db_engine: sqlalchemy.engine.Engine,
    interval_name: str,
    id_val: int,
) -> None:
    os.environ["PARTITION_INTERVAL"] = interval_name
    server.py.framework.db.sqldb.models.Base.metadata.create_all(db_engine)

    current_time: datetime = datetime.now(UTC)
    activation: server.py.framework.db.sqldb.models.AlertActivation = (
        server.py.framework.db.sqldb.models.AlertActivation(
            id=id_val,
            activation_time=current_time,
            project="project_a",
            name="alert_b",
            entity_id="entity_2",
            entity_kind="kind_2",
            event_kind="event_2",
            severity="low",
            number_of_events=2,
        )
    )

    with sqlalchemy.orm.Session(db_engine) as session:
        session.add(activation)
        session.flush()  # fires before_insert listener
        interval = mlrun.common.schemas.partition_interval.PartitionInterval(
            interval_name
        )
        expected_key = interval.get_partition_key_value(
            current_datetime=current_time,
        )
        assert activation.partition_key == expected_key
