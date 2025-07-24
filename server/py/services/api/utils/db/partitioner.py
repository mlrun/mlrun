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

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

import mlrun
from mlrun.common.schemas import PartitionInterval

import framework.db.sqldb.db
import framework.utils.singletons.db
from framework.db.sqldb.partititioner import RangePartitioner


class DBPartitioner:
    """
    Manages creation and dropping of range partitions based on retention policy.
    """

    def __init__(
        self,
        buffer_multiplier_override: float = None,
    ):
        self._buffer_multiplier = (
            buffer_multiplier_override or mlrun.mlconf.partitions_buffer_multiplier
        )

    def create_partitions(
        self,
        session: Session,
        table_name: str,
        retention_days: int,
        partition_interval: PartitionInterval,
    ) -> None:
        # compute total partitions needed (retention + buffer)
        total_days = (
            retention_days
            + self._buffer_multiplier * partition_interval.as_duration().days
        )
        partition_count = partition_interval.get_number_of_partitions(
            days=int(total_days)
        )

        # create or extend partitions
        partitioner = RangePartitioner(session.get_bind().dialect.name)
        partitioner.apply_partitions(session, table_name, partition_count)
        session.flush()

    def create_and_drop_partitions(
        self,
        session: Session,
        table_name: str,
        retention_days: int,
    ) -> None:
        """
        Ensure future partitions for retention + buffer, and drop expired ones.

        :param session: SQLAlchemy session for database operations.
        :param table_name: Name of the table to manage partitions for.
        :param retention_days: Number of days to retain partitions.
        """
        # determine the existing partition interval
        partition_interval = self.get_partition_interval(session, table_name)

        self.create_partitions(
            session=session,
            table_name=table_name,
            partition_interval=partition_interval,
        )

        # drop partitions older than retention
        self.drop_partitions(
            session=session,
            table_name=table_name,
            partition_interval=partition_interval,
            retention_days=retention_days,
        )
        session.flush()

    def get_partition_interval(
        self,
        session: Session,
        table_name: str,
    ) -> PartitionInterval:
        db_client = framework.utils.singletons.db.get_db()
        partition_expr = db_client.get_partition_expression_for_table(
            session=session,
            table_name=table_name,
        )
        if not partition_expr:
            if db_client.table_exists(session=session, table_name=table_name):
                reason = "Table is not partitioned"
            else:
                reason = "Table does not exist"
            raise ValueError(
                f"Cannot determine partition interval for '{table_name}': {reason}"
            )
        return PartitionInterval.from_expression(partition_expr)

    def drop_partitions(
        self,
        session: Session,
        table_name: str,
        partition_interval: PartitionInterval,
        retention_days: int,
    ) -> None:
        db_client = framework.utils.singletons.db.get_db()
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
        cutoff_partition_name = f"p{partition_interval.get_partition_name(cutoff_date)}"
        db_client.drop_partitions(
            session=session,
            table_name=table_name,
            cutoff_partition_name=cutoff_partition_name,
        )
