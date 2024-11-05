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

import mlrun.common.schemas.alert
import mlrun.utils.singleton
import services.api.db.sqldb.db


class AlertActivation(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def create_and_drop_partitions(
        self,
        session: Session,
        retention_days: int,
    ) -> None:
        """
        Creates partitions for the future based on the specified retention days
        and drops old partitions that are older than the retention period.

        :param session: SQLAlchemy session for database operations.
        :param retention_days: The number of days to retain partitions.
        """

        # Retrieve the partition expression and interval from the database.
        partition_expression, partition_interval = (
            self.get_partition_expression_and_interval(session)
        )

        # Ensure partitions for double the retention time.
        partition_number = partition_interval.get_number_of_partitions(
            days=2 * retention_days
        )

        # Create the calculated number of partitions.
        self.create_partitions(session, partition_number)

        # Drop old partitions that exceed the retention period.
        self.drop_partitions(session, retention_days)

    def create_partitions(
        self,
        session: Session,
        partition_number: int,
    ):
        # Retrieve the partition function from the database
        partition_expression, partition_interval = (
            self.get_partition_expression_and_interval(session)
        )

        partitioning_information_list = []
        current_datetime = datetime.now()

        for _ in range(partition_number):
            partitioning_information_list.append(
                partition_interval.get_partition_info(current_datetime)
            )

            # Move to the next interval based on the partition_interval
            current_datetime = mlrun.common.schemas.alert.PartitionInterval(
                partition_interval
            ).get_next_partition_time(current_datetime=current_datetime)

        services.api.utils.singletons.db.get_db().create_partitions(
            session=session,
            table_name=self.table_name,
            partitioning_information_list=partitioning_information_list,
        )

    def drop_partitions(
        self,
        session: Session,
        retention_days: int,
    ):
        """
        Drop partitions older than the specified retention period.

        :param session: SQLAlchemy session.
        :param retention_days: Retention period in days.
        """
        _, partition_interval = self.get_partition_expression_and_interval(session)

        # Calculate the cutoff date for partition retention
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Generate cutoff partition name based on the interval
        cutoff_partition_name = partition_interval.get_partition_name(cutoff_date)

        # Drop partitions that are older than the cutoff
        services.api.utils.singletons.db.get_db().drop_partitions(
            session,
            self.table_name,
            f"p{cutoff_partition_name}",
        )

    def get_partition_expression_and_interval(
        self, session: Session
    ) -> tuple[str, mlrun.common.schemas.alert.PartitionInterval]:
        # Retrieve the partition function from the database
        partition_expression = services.api.utils.singletons.db.get_db().get_partition_expression_for_table(
            session,
            table_name=self.table_name,
        )

        partition_function = partition_expression[
            : partition_expression.find("(")
        ].upper()
        partition_interval = mlrun.common.schemas.alert.PartitionInterval.from_function(
            partition_function
        )
        return partition_expression, partition_interval

    @property
    def table_name(self) -> str:
        return "alert_activation"
