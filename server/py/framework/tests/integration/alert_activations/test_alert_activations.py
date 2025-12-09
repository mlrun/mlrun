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
from datetime import datetime

import pytest
import sqlalchemy.engine
import sqlalchemy.orm

import mlrun.common.db.dialects
import mlrun.common.schemas as schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.notification as notification_objects
import mlrun.common.schemas.partition_interval
import server.py.framework.db.sqldb.db
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

    db = server.py.framework.db.sqldb.db.SQLDB(
        dsn=db_engine.url.render_as_string(
            hide_password=False,
        )
    )
    with sqlalchemy.orm.Session(db_engine) as session:
        current_time: datetime = datetime.now()

        event_entities = alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
            project="project_a",
            ids=["entity_2"],
        )

        alert_config_data = schemas.AlertConfig(
            project="project_a",
            name="alert_b",
            description="test alert for store_alert_activation",
            summary="test summary",
            severity=alert_objects.AlertSeverity.LOW,
            entities=event_entities,
            trigger=alert_objects.AlertTrigger(
                events=[alert_objects.EventKind.FAILED],
                prometheus_alert="test",
            ),
            criteria=alert_objects.AlertCriteria(
                count=2,
                period="1d",
            ),
            reset_policy=alert_objects.ResetPolicy.AUTO,
            notifications=[
                alert_objects.AlertNotification(
                    notification=notification_objects.Notification(
                        kind=notification_objects.NotificationKind.slack,
                        name="test-slack",
                        message="Test alert",
                        severity=notification_objects.NotificationSeverity.INFO,
                        when=["completed"],
                        params={"webhook": "https://example.com/hook"},
                        status=notification_objects.NotificationStatus.PENDING,
                    ),
                    cooldown_period="1d",
                )
            ],
            state=alert_objects.AlertActiveState.INACTIVE,
            count=0,
        )

        event_data_object: schemas.Event = schemas.Event(
            kind=alert_objects.EventKind.FAILED,
            timestamp=current_time,
            entity=event_entities,
            value_dict={},
        )

        alert_activation_id = db.store_alert_activation(
            session=session,
            alert_data=alert_config_data,
            event_data=event_data_object,
        )
        stored = (
            session.query(server.py.framework.db.sqldb.models.AlertActivation)
            .filter(
                server.py.framework.db.sqldb.models.AlertActivation.id
                == alert_activation_id
            )
            .one()
        )

        interval = mlrun.common.schemas.partition_interval.PartitionInterval(
            interval_name
        )
        expected_key = interval.get_partition_key_value(
            current_datetime=current_time,
        )
        assert stored.partition_key == expected_key
