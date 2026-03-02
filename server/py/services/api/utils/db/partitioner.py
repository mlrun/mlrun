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
import math
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

import mlrun
import mlrun.common.schemas
import mlrun.common.schemas.partition_interval

import framework.db.sqldb.db
import framework.db.sqldb.partition_bootstrapper
import framework.utils.singletons.db


class DBPartitioner:
    def __init__(
        self,
        buffer_multiplier_override: float | None = None,
    ):
        """
        Initialize the partition manager.

        :param buffer_multiplier_override:
            Fractional buffer applied when pre-creating partitions
            (e.g. 0.25 = 25% extra). A float is used to allow fractional
            over-provisioning; the final count is rounded up safely.
        """
        self._buffer_multiplier = (
            mlrun.mlconf.partitions_buffer_multiplier
            if buffer_multiplier_override is None
            else float(buffer_multiplier_override)
        )
        self._db = framework.utils.singletons.db.get_db()

    def create_and_drop_partitions(
        self,
        session: Session,
        table_name: str,
        retention_days: int,
        partitions_to_create: int = 1,
    ) -> None:
        """
        Ensure future partitions for retention + buffer, and drop expired ones.

        :param session: SQLAlchemy session for database operations.
        :param table_name: Name of the table to manage partitions for.
        :param retention_days: Number of days to retain partitions.
        :param partitions_to_create: Number of partitions to create, defaults to 1.
        """
        # determine the existing partition interval
        partition_interval = self.get_partition_interval(
            session=session,
            table_name=table_name,
        )

        self.create_partitions(
            session=session,
            table_name=table_name,
            partition_interval=partition_interval,
            partitions_to_create=partitions_to_create,
        )

        # drop partitions older than retention
        self.drop_partitions(
            session=session,
            table_name=table_name,
            partition_interval=partition_interval,
            retention_days=retention_days,
        )
        # Flush is required to force execution of partition DDL immediately,
        # ensuring newly created partitions are visible to subsequent operations
        # in the same session (this is not a commit).
        session.flush()

    def create_partitions(
        self,
        session: Session,
        table_name: str,
        partitions_to_create: int,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
    ) -> None:
        """
        Create future partitions for a table, including buffer.

        :param session: SQLAlchemy session for database operations.
        :param table_name: Name of the table to manage partitions for.
        :param partitions_to_create: Number of partitions to create before buffering.
        :param partition_interval: Partition interval configured for the table.
        """
        partitions_count = max(
            1, math.ceil(partitions_to_create * (1 + self._buffer_multiplier))
        )

        partitioner = framework.db.sqldb.partition_bootstrapper.PartitionBootstrapper(
            session.get_bind().dialect.name
        )
        partitioner.bootstrap(
            session=session,
            table_name=table_name,
            partition_interval=partition_interval,
            partitions_count=partitions_count,
        )
        session.flush()

    def get_partition_interval(
        self, session: Session, table_name: str
    ) -> mlrun.common.schemas.partition_interval.PartitionInterval:
        """
        Retrieve the partition interval configured for a table.

        :param session: SQLAlchemy session for database operations.
        :param table_name: Name of the table to look up.
        :return: The configured partition interval.
        """
        return self._db.get_partition_interval_for_table(session, table_name)

    def drop_partitions(
        self,
        session: Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        retention_days: int,
    ) -> None:
        """
        Drop partitions older than the retention window.

        :param session: SQLAlchemy session for database operations.
        :param table_name: Name of the table to manage partitions for.
        :param partition_interval: Partition interval configured for the table.
        :param retention_days: Number of days to retain partitions.
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
        self._db.drop_partitions(
            session=session,
            table_name=table_name,
            cutoff_partition_name=partition_interval.get_partition_name(cutoff_date),
        )
