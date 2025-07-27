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
import abc

try:
    from datetime import UTC, datetime  # UTC is only defined in Python 3.11+
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.compiler

import mlrun
import mlrun.common.db.dialects
import mlrun.common.schemas.partition_interval
import mlrun.utils


class PartitionBootstrapper:
    def __new__(cls, dialect: str):
        if dialect.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
            return super().__new__(PartitionBootstrapperMySQL)
        elif dialect.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL):
            return super().__new__(PartitionBootstrapperPostgres)
        elif dialect.startswith(mlrun.common.db.dialects.Dialects.SQLITE):
            return super().__new__(PartitionBootstrapperSqlite)
        raise ValueError(f"Unsupported dialect: {dialect}")

    @abc.abstractmethod
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        raise NotImplementedError()

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

    def _get_partition_names_and_boundaries(
        self,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ) -> list[tuple[str, int]]:
        return partition_interval.get_partition_names_and_boundaries(
            start_datetime=datetime.now(UTC),
            partitions_count=partitions_count,
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


class PartitionBootstrapperMySQL(PartitionBootstrapper):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        partition_list = self._get_partition_names_and_boundaries(
            partitions_count=partitions_count,
            partition_interval=partition_interval,
        )
        if not partition_list:
            mlrun.utils.logger.warning(
                "No partitions to create for table",
                table_name=table_name,
            )
            return

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
        mlrun.utils.logger.info(
            "Creating partitions",
            partitions_count=len(partition_list),
            table_name=table_name,
        )
        session.execute(sqlalchemy.text(ddl))
        session.commit()


class PartitionBootstrapperPostgres(PartitionBootstrapper):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        partition_list = self._get_partition_names_and_boundaries(
            partition_interval=partition_interval,
            partitions_count=partitions_count,
        )
        if not partition_list:
            mlrun.utils.logger.warning(
                "No partitions to create for table",
                table_name=table_name,
            )
            return

        quoted_table = self._quote_table_name(
            session=session,
            table_name=table_name,
            sample_partition=partition_list[0][0],
        )

        mlrun.utils.logger.info(
            "Creating partitions",
            partitions_count=len(partition_list),
            table_name=table_name,
        )

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


class PartitionBootstrapperSqlite(PartitionBootstrapper):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        mlrun.utils.logger.info(
            "SQLite does not support partitioning natively, skipping bootstrap."
        )
