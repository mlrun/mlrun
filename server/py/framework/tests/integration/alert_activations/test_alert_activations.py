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
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

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
import server.py.services.api.utils.db.partitioner
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


def store_alert_activation_at_timestamp(
    database: server.py.framework.db.sqldb.db.SQLDB,
    database_session: sqlalchemy.orm.Session,
    activation_timestamp: datetime,
    index_for_timestamp: int,
    activation_index_within_day: int,
) -> None:
    event_entities = mlrun.common.schemas.alert.EventEntities(
        kind=mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
        project="project_a",
        ids=["entity_2"],
    )

    alert_config_data = mlrun.common.schemas.AlertConfig(
        project="project_a",
        name=f"alert_{index_for_timestamp}_{activation_index_within_day}",
        description="test alert for retention",
        summary="test summary",
        severity=mlrun.common.schemas.alert.AlertSeverity.LOW,
        entities=event_entities,
        trigger=mlrun.common.schemas.alert.AlertTrigger(
            events=[mlrun.common.schemas.alert.EventKind.FAILED],
            prometheus_alert="test",
        ),
        criteria=mlrun.common.schemas.alert.AlertCriteria(
            count=2,
            period="1d",
        ),
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.AUTO,
        notifications=[
            mlrun.common.schemas.alert.AlertNotification(
                notification=mlrun.common.schemas.notification.Notification(
                    kind=mlrun.common.schemas.notification.NotificationKind.slack,
                    name="test-slack",
                    params={"webhook": "https://example.com/fake-webhook"},
                ),
                cooldown_period="1h",
            )
        ],
        state=mlrun.common.schemas.alert.AlertActiveState.INACTIVE,
        count=0,
    )

    event_data_object: mlrun.common.schemas.Event = mlrun.common.schemas.Event(
        kind=mlrun.common.schemas.alert.EventKind.FAILED,
        timestamp=activation_timestamp,
        entity=event_entities,
        value_dict={},
    )

    database.store_alert_activation(
        session=database_session,
        alert_data=alert_config_data,
        event_data=event_data_object,
    )


def get_alert_activations_partition_statistics(
    database_session: sqlalchemy.orm.Session,
) -> list[Mapping[str, Any]]:
    """
    Return MySQL partition metadata for alert_activations.

    Keys:
      PARTITION_NAME, PARTITION_ORDINAL_POSITION, PARTITION_METHOD,
      boundary_value, TABLE_ROWS
    """
    database_session.execute(sqlalchemy.text("ANALYZE TABLE alert_activations"))
    database_session.commit()

    result = database_session.execute(
        sqlalchemy.text(
            "SELECT "
            "partition_name, "
            "partition_ordinal_position, "
            "partition_method, "
            "partition_description AS boundary_value, "
            "table_rows "
            "FROM information_schema.PARTITIONS "
            "WHERE table_schema = DATABASE() "
            "AND table_name = 'alert_activations' "
            "ORDER BY partition_ordinal_position"
        ),
    )

    return list(result.mappings())


def build_partition_ranges(
    partition_statistics: list[Mapping[str, Any]],
) -> list[tuple[str, int | None, int]]:
    """
    Build (partition_name, lower_bound, upper_bound) from MySQL metadata.
    """
    partition_ranges = []
    previous_upper_bound = None

    for partition_row in partition_statistics:
        partition_name = partition_row["PARTITION_NAME"]
        partition_upper_bound = int(partition_row["boundary_value"])
        partition_ranges.append(
            (partition_name, previous_upper_bound, partition_upper_bound),
        )
        previous_upper_bound = partition_upper_bound

    return partition_ranges


def map_rows_to_partitions(
    alert_activations: list[server.py.framework.db.sqldb.models.AlertActivation],
    partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
    partition_ranges: list[tuple[str, int | None, int]],
) -> dict[str, list[server.py.framework.db.sqldb.models.AlertActivation]]:
    """
    Assert each row's partition_key matches activation_time and falls into one range.
    """
    rows_by_partition_name = {name: [] for name, _, _ in partition_ranges}

    for alert_activation in alert_activations:
        partition_key = alert_activation.partition_key
        expected_key = partition_interval.get_partition_key_value(
            current_datetime=alert_activation.activation_time,
        )
        # Sanity: key matches activation_time
        assert partition_key == expected_key

        matched_partition_name = None
        for partition_name, lower_bound, upper_bound in partition_ranges:
            lower_ok = lower_bound is None or partition_key >= lower_bound
            upper_ok = partition_key < upper_bound
            if lower_ok and upper_ok:
                matched_partition_name = partition_name
                rows_by_partition_name[partition_name].append(alert_activation)
                break

        # Every row must land in exactly one partition range
        assert matched_partition_name is not None

    return rows_by_partition_name


@pytest.mark.integration
@tests.common_fixtures.freeze_datetime(datetime(2025, 1, 12))
def test_drop_partitions_drops_old_rows(
    db_engine: sqlalchemy.engine.Engine,
) -> None:
    """
    Insert 3 days of data (2 rows per day) so each day lands in its own partition.
    After retention, only the newest day's rows should remain in a single partition.
    """
    os.environ["PARTITION_INTERVAL"] = "DAY"
    server.py.framework.db.sqldb.models.Base.metadata.create_all(db_engine)

    database = server.py.framework.db.sqldb.db.SQLDB(
        dsn=db_engine.url.render_as_string(hide_password=False),
    )

    number_of_days_to_insert = 3
    number_of_alert_activations_per_day = 2
    base_time = datetime(2025, 1, 12)
    partition_interval = mlrun.common.schemas.partition_interval.PartitionInterval(
        "DAY"
    )

    with sqlalchemy.orm.Session(db_engine) as database_session:
        partitioner = server.py.services.api.utils.db.partitioner.DBPartitioner(
            buffer_multiplier_override=0,
        )

        # Pre-create partitions so upcoming days each get their own RANGE partition.
        partitioner.create_and_drop_partitions(
            session=database_session,
            table_name="alert_activations",
            retention_days=0,
            partitions_to_create=4,
        )

        # Insert 2 rows per day for 3 consecutive days:
        # 2025-01-12, 2025-01-13, 2025-01-14.
        for day_offset in range(number_of_days_to_insert):
            current_timestamp_for_day = base_time + timedelta(days=day_offset)
            for activation_index_within_day in range(
                number_of_alert_activations_per_day,
            ):
                store_alert_activation_at_timestamp(
                    database=database,
                    database_session=database_session,
                    activation_timestamp=current_timestamp_for_day,
                    index_for_timestamp=day_offset,
                    activation_index_within_day=activation_index_within_day,
                )

        database_session.flush()

        # 3 days × 2 rows = 6 total.
        all_before_drop = (
            database_session.query(server.py.framework.db.sqldb.models.AlertActivation)
            .order_by(
                server.py.framework.db.sqldb.models.AlertActivation.activation_time,
                server.py.framework.db.sqldb.models.AlertActivation.id,
            )
            .all()
        )
        pre_drop_activation_count = len(all_before_drop)
        assert pre_drop_activation_count == 6

        # Partition layout and row placement before retention.
        partition_statistics_before = get_alert_activations_partition_statistics(
            database_session=database_session,
        )
        partition_ranges_before = build_partition_ranges(partition_statistics_before)
        rows_by_partition_name_before = map_rows_to_partitions(
            all_before_drop,
            partition_interval,
            partition_ranges_before,
        )

        # Expect 3 non-empty partitions, each with 2 rows.
        non_empty_partitions_before = [
            name for name, rows in rows_by_partition_name_before.items() if rows
        ]
        assert len(non_empty_partitions_before) == 3
        for name in non_empty_partitions_before:
            assert len(rows_by_partition_name_before[name]) == 2

        # Move "now" forward; retention should keep only the newest day.
        tests.common_fixtures.FrozenDatetime._frozen_now = datetime(
            2025,
            1,
            15,
        )

        partitioner.create_and_drop_partitions(
            session=database_session,
            table_name="alert_activations",
            retention_days=1,
            partitions_to_create=3,
        )

        # After retention: only the newest day's 2 rows should remain.
        remaining_alert_activations = (
            database_session.query(server.py.framework.db.sqldb.models.AlertActivation)
            .order_by(
                server.py.framework.db.sqldb.models.AlertActivation.activation_time,
                server.py.framework.db.sqldb.models.AlertActivation.id,
            )
            .all()
        )

        remaining_activation_count = len(remaining_alert_activations)
        assert remaining_activation_count == 2

        # All remaining rows should belong to the latest day.
        remaining_dates = {
            alert_activation.activation_time.date()
            for alert_activation in remaining_alert_activations
        }
        expected_latest_date = (base_time + timedelta(days=2)).date()  # 2025-01-14
        assert remaining_dates == {expected_latest_date}

        remaining_partition_keys = {
            alert_activation.partition_key
            for alert_activation in remaining_alert_activations
        }
        assert len(remaining_partition_keys) == 1
        latest_partition_key = next(iter(remaining_partition_keys))

        # Partition layout and row placement after retention.
        partition_statistics_after = get_alert_activations_partition_statistics(
            database_session=database_session,
        )
        partition_ranges_after = build_partition_ranges(partition_statistics_after)
        rows_by_partition_name_after = map_rows_to_partitions(
            remaining_alert_activations,
            partition_interval,
            partition_ranges_after,
        )

        # Exactly one partition should be non-empty.
        non_empty_partition_names_after = {
            name for name, rows in rows_by_partition_name_after.items() if rows
        }
        assert len(non_empty_partition_names_after) == 1
        single_non_empty_partition_name = next(iter(non_empty_partition_names_after))

        # That single partition must contain the latest partition_key.
        for (
            partition_name,
            partition_lower_bound,
            partition_upper_bound,
        ) in partition_ranges_after:
            if partition_name != single_non_empty_partition_name:
                continue

            lower_ok_for_latest = (
                partition_lower_bound is None
                or latest_partition_key >= partition_lower_bound
            )
            upper_ok_for_latest = latest_partition_key < partition_upper_bound
            assert lower_ok_for_latest and upper_ok_for_latest
            break
