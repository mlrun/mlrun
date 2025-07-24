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

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.compiler

import mlrun.common.db.dialects
import mlrun.common.schemas.partition


class RangePartitioner:
    def __new__(cls, dialect: str):
        if dialect.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
            return super().__new__(RangePartitionerMySQL)
        if dialect.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL):
            return super().__new__(RangePartitionerPostgres)
        raise ValueError(f"Unsupported dialect: {dialect}")

    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        raise NotImplementedError

    def get_quoted_partitioned_table_params(
        self,
        partition_name: str,
        session: sqlalchemy.orm.Session,
        table_name: str,
    ) -> tuple[str, str]:
        preparer = sqlalchemy.sql.compiler.IdentifierPreparer(
            session.get_bind().dialect
        )
        quoted_table = preparer.quote(table_name)
        quoted_partition = preparer.quote(partition_name)
        return quoted_partition, quoted_table

    def _compute_partitions(self, num_partitions: int = 2) -> list[tuple[str, str]]:
        interval_key = os.getenv(
            "PARTITION_INTERVAL",
            mlrun.common.schemas.partition.PartitionInterval.YEARWEEK.value,
        ).upper()
        interval = mlrun.common.schemas.partition.PartitionInterval(interval_key)
        return interval.get_partition_info(
            datetime.now(UTC), partition_number=num_partitions
        )

    def _quote_table_name(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        sample_partition: str,
    ) -> str:
        _, quoted_table = self.get_quoted_partitioned_table_params(
            partition_name=sample_partition,
            session=session,
            table_name=table_name,
        )
        return quoted_table


class RangePartitionerMySQL(RangePartitioner):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        # prepare current and next partition definitions
        partition_list = self._compute_partitions(num_partitions=2)
        quoted_table = self._quote_table_name(session, table_name, partition_list[0][0])

        partition_clauses = []
        for partition_name, boundary_value in partition_list:
            quoted_partition, _ = self.get_quoted_partitioned_table_params(
                partition_name=partition_name,
                session=session,
                table_name=table_name,
            )
            partition_clauses.append(
                f"PARTITION {quoted_partition} VALUES LESS THAN ({int(boundary_value)})"
            )
        join_str = ",\n"
        ddl = f"""
            ALTER TABLE {quoted_table}
            PARTITION BY RANGE (partition_key) (
                {join_str.join(partition_clauses)}
            )
            """
        session.execute(sqlalchemy.text(ddl))
        session.commit()


class RangePartitionerPostgres(RangePartitioner):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        # prepare current and next partition definitions
        partition_list = self._compute_partitions(num_partitions=2)
        quoted_table = self._quote_table_name(session, table_name, partition_list[0][0])

        for index, (partition_name, boundary_value) in enumerate(partition_list):
            quoted_partition, _ = self.get_quoted_partitioned_table_params(
                partition_name=partition_name,
                session=session,
                table_name=table_name,
            )
            lower_bound = (
                "MINVALUE" if index == 0 else str(int(partition_list[index - 1][1]))
            )
            upper_bound = str(int(boundary_value))
            ddl = f"""
                CREATE TABLE {quoted_partition}
                PARTITION OF {quoted_table}
                FOR VALUES FROM ({lower_bound}) TO ({upper_bound})
            """
            session.execute(sqlalchemy.text(ddl))
        session.commit()
