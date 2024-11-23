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

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

import mlrun.common.schemas.partition
import mlrun.config

import framework.db.sqldb.db
import framework.utils.singletons.db


class MySQLPartitioner:
    def create_and_drop_partitions(
        self,
        session: Session,
        table_name: str,
        retention_days: int,
    ) -> None:
        """
        Creates partitions for the future based on the specified retention days
        and drops old partitions that are older than the retention period.

        :param session: SQLAlchemy session for database operations.
        :param table_name: Name of the table to create/drop partitions.
        :param retention_days: The number of days to retain partitions.

        """

        # Retrieve the partition interval from the database
        partition_interval = self.get_partition_interval(
            session,
            table_name,
        )

        # Ensure partitions for the retention time plus a buffer
        partition_number = partition_interval.get_number_of_partitions(
            days=retention_days
            + mlrun.mlconf.partitions_buffer_multiplier
            * partition_interval.as_duration().days
        )

        # Create the calculated number of partitions.
        self.create_partitions(
            session, table_name, partition_number, partition_interval
        )

        # Drop old partitions that exceed the retention period.
        self.drop_partitions(session, table_name, retention_days, partition_interval)

    @staticmethod
    def create_partitions(
        session: Session,
        table_name: str,
        partition_number: int,
        partition_interval: mlrun.common.schemas.partition.PartitionInterval,
    ):
        partitioning_information_list = partition_interval.get_partition_info(
            start_datetime=datetime.now(),
            partition_number=partition_number,
        )

        framework.utils.singletons.db.get_db().create_partitions(
            session=session,
            table_name=table_name,
            partitioning_information_list=partitioning_information_list,
        )

    @staticmethod
    def drop_partitions(
        session: Session,
        table_name: str,
        retention_days: int,
        partition_interval: mlrun.common.schemas.partition.PartitionInterval,
    ):
        """
        Drop partitions older than the specified retention period.

        :param session: SQLAlchemy session.
        :param table_name: Name of the table to drop partitions from.
        :param retention_days: Retention period in days.
        :param partition_interval: Partition interval
        """
        # Calculate the cutoff date for partition retention
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Generate cutoff partition name based on the interval
        cutoff_partition_name = partition_interval.get_partition_name(cutoff_date)

        # Drop partitions that are older than the cutoff
        framework.utils.singletons.db.get_db().drop_partitions(
            session,
            table_name,
            f"p{cutoff_partition_name}",
        )

    @staticmethod
    def get_partition_interval(
        session: Session,
        table_name: str,
    ) -> mlrun.common.schemas.partition.PartitionInterval:
        # Retrieve the partition function from the database
        partition_expression = (
            framework.utils.singletons.db.get_db().get_partition_expression_for_table(
                session,
                table_name=table_name,
            )
        )

        partition_function = partition_expression[
            : partition_expression.find("(")
        ].upper()
        partition_interval = (
            mlrun.common.schemas.partition.PartitionInterval.from_function(
                partition_function
            )
        )
        return partition_interval
