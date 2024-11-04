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
    @staticmethod
    def get_partition_info(
        partition_interval: str,
        partition_datetime: datetime,
    ) -> tuple[str, str, str]:
        """
        Generates partition details for a specified interval and datetime.

        :param partition_interval: The partitioning interval type, e.g., "DAY", "MONTH", or "YEARWEEK".
        :param partition_datetime: The datetime used for generating partition details.

        :return: A tuple containing:
            - partition_name: The name for the partition.
            - partition_value: The "LESS THAN" value for the next partition boundary.
            - partition_expression: The SQL partition expression.
        """
        if partition_interval == mlrun.common.schemas.alert.PartitionInterval.YEARWEEK:
            year, week, _ = partition_datetime.isocalendar()
            partition_name = f"{year}{week:02d}"

            next_week = partition_datetime + timedelta(weeks=1)
            next_year, next_week_num, _ = next_week.isocalendar()
            partition_value = f"{next_year}{next_week_num:02d}"

            partition_expression = "YEARWEEK(activation_time, 1)"
            return partition_name, partition_value, partition_expression

        if partition_interval == mlrun.common.schemas.alert.PartitionInterval.DAY:
            partition_name = partition_datetime.strftime("%Y%m%d")
            partition_boundary_date = partition_datetime + timedelta(days=1)
            # Format as 'YYYYMMDD' (year, month, day)
            partition_value = partition_boundary_date.strftime("%Y%m%d")
        elif partition_interval == mlrun.common.schemas.alert.PartitionInterval.MONTH:
            partition_name = partition_datetime.strftime("%Y%m")
            partition_boundary_date = (
                partition_datetime.replace(day=1) + timedelta(days=32)
            ).replace(day=1)
            # Format as 'YYYYMM' (year and month)
            partition_value = partition_boundary_date.strftime("%Y%m")
        else:
            raise ValueError(f"Unsupported partition interval: {partition_interval}")
        partition_expression = f"{partition_interval}(activation_time)"

        return partition_name, partition_value, partition_expression

    def create_and_drop_partitions(
        self,
        session: Session,
        retention_weeks: int,
    ) -> None:
        """
        Creates partitions for the future based on the specified retention weeks
        and drops old partitions that are older than the retention period.

        :param session: SQLAlchemy session for database operations.
        :param retention_weeks: The number of weeks to retain partitions.
        """

        # Retrieve the partition expression and interval from the database.
        partition_expression, partition_interval = (
            self.get_partition_expression_and_interval(session)
        )

        # Ensure partitions for double the retention time.
        ensure_partitions_weeks = 2 * retention_weeks

        # Calculate the number of future partitions to create based on the partition interval.
        if partition_interval == "DAY":
            partition_number = ensure_partitions_weeks * 7
        elif partition_interval == "MONTH":
            # Average number days in a month is 30.44
            partition_number = int(ensure_partitions_weeks * 7 / 30.44)
        elif partition_interval == "YEARWEEK":
            partition_number = ensure_partitions_weeks
        else:
            raise ValueError(f"Unsupported partition interval: {partition_interval}")

        # Create the calculated number of partitions.
        self.create_partitions(session, partition_number)

        # Drop old partitions that exceed the retention period.
        self.drop_old_partitions(session, retention_weeks)

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
            # Generate partition information for the current interval
            partition_name, partition_value, partition_expression = (
                self.get_partition_info(partition_interval, current_datetime)
            )
            partitioning_information_list.append(
                (partition_name, partition_value, partition_expression)
            )

            # Move to the next interval based on the partition_interval
            if partition_interval == "DAY":
                current_datetime += timedelta(days=1)
            elif partition_interval == "MONTH":
                current_datetime = (
                    current_datetime.replace(day=1) + timedelta(days=32)
                ).replace(day=1)
            elif partition_interval == "YEARWEEK":
                current_datetime += timedelta(weeks=1)

        services.api.utils.singletons.db.get_db().create_partitions(
            session=session,
            table_name=self.table_name,
            partitioning_information_list=partitioning_information_list,
        )

    def drop_old_partitions(
        self,
        session: Session,
        retention_weeks: int,
    ):
        """
        Drop partitions older than the specified retention period.

        :param session: SQLAlchemy session.
        :param retention_weeks: Retention period in weeks.
        """
        _, partition_interval = self.get_partition_expression_and_interval(session)

        # Calculate the cutoff date for partition retention
        cutoff_date = datetime.now() - timedelta(weeks=retention_weeks)

        # Generate cutoff partition name based on the interval
        cutoff_partition_name = self.generate_cutoff_partition_name(
            cutoff_date, partition_interval
        )

        # Drop partitions that are older than the cutoff
        services.api.utils.singletons.db.get_db().drop_partitions(
            session,
            self.table_name,
            cutoff_partition_name,
        )

    @staticmethod
    def generate_cutoff_partition_name(
        cutoff_date: datetime,
        partition_interval: str,
    ) -> str:
        """
        Generate the cutoff partition name based on the cutoff date and interval.

        :param cutoff_date: The date used to determine the partition cutoff.
        :param partition_interval: The partition interval type, e.g., "DAY", "MONTH", "YEARWEEK".

        :return: The cutoff partition name.
        """
        if partition_interval == "DAY":
            return f"p{cutoff_date.strftime('%Y%m%d')}"
        elif partition_interval == "MONTH":
            return f"p{cutoff_date.strftime('%Y%m')}"
        elif partition_interval == "YEARWEEK":
            year, week, _ = cutoff_date.isocalendar()
            return f"p{year}{week:02d}"
        else:
            raise ValueError("Unsupported partition interval")

    def get_partition_expression_and_interval(self, session: Session):
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
