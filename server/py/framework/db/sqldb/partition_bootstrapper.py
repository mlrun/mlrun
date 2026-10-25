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
from datetime import UTC, datetime

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.compiler

import mlrun
import mlrun.common.db.dialects
import mlrun.common.schemas.partition_interval
import mlrun.utils


class PartitionBootstrapper:
    def __new__(cls, dialect: str):
        """
        Factory that returns a dialect-specific bootstrapper instance.
        """
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
        """
        Ensure the table has the required partitions for the given dialect.
        """
        raise NotImplementedError()

    def get_quoted_partitioned_table_params(
        self,
        partition_name: str,
        session: sqlalchemy.orm.Session,
        table_name: str,
    ) -> tuple[str, str]:
        """
        Return safely quoted table and partition identifiers for the current dialect.
        """
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
        """
        Compute target partition names and upper bounds from the interval definition.
        """
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
        """
        Quote a table name using the dialect-specific identifier rules.
        """
        _, quoted_table = self.get_quoted_partitioned_table_params(
            partition_name=sample_partition,
            session=session,
            table_name=table_name,
        )
        return quoted_table

    def _get_partition_list(
        self,
        *,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ) -> list[tuple[str, int]]:
        """
        Build the desired partition list and log if no partitions are requested.
        """
        partition_list = self._get_partition_names_and_boundaries(
            partition_interval=partition_interval,
            partitions_count=partitions_count,
        )
        if not partition_list:
            mlrun.utils.logger.warning(
                "No partitions to create for table",
                table_name=table_name,
            )
        return partition_list


class PartitionBootstrapperMySQL(PartitionBootstrapper):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        """
        Initialize or extend MySQL RANGE partitions without reorganizing existing data.
        """
        partition_list = self._get_partition_list(
            table_name=table_name,
            partition_interval=partition_interval,
            partitions_count=partitions_count,
        )
        if not partition_list:
            return

        quoted_table = self._quote_table_name(
            session=session,
            table_name=table_name,
            sample_partition=partition_list[0][0],
        )

        existing_partitions = self._get_existing_partition_boundaries(
            session=session,
            table_name=table_name,
        )

        if not existing_partitions:
            self._create_initial_partitions(
                session=session,
                table_name=table_name,
                quoted_table=quoted_table,
                partition_list=partition_list,
            )
            return

        self._add_new_partitions(
            session=session,
            table_name=table_name,
            quoted_table=quoted_table,
            partition_list=partition_list,
            existing_partitions=existing_partitions,
        )

    def _get_existing_partition_boundaries(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
    ) -> list[tuple[str, int]]:
        """
        Read existing MySQL partitions and their numeric VALUES LESS THAN bounds.
        """
        sql = sqlalchemy.text(
            """
            SELECT PARTITION_NAME, PARTITION_DESCRIPTION
            FROM information_schema.PARTITIONS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table_name
            ORDER BY PARTITION_DESCRIPTION
            """
        )
        result = session.execute(sql, {"table_name": table_name})
        existing_partitions = []
        for partition_name, partition_description in result:
            if partition_description is None:
                continue
            try:
                boundary_value = int(partition_description)
            except (TypeError, ValueError):
                continue
            existing_partitions.append((partition_name, boundary_value))
        return existing_partitions

    def _create_initial_partitions(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        quoted_table: str,
        partition_list: list[tuple[str, int]],
    ) -> None:
        """
        Create the initial RANGE partitioning for an unpartitioned MySQL table.
        """
        partition_clauses = []
        for partition_name, boundary_value in partition_list:
            quoted_partition, _ = self.get_quoted_partitioned_table_params(
                partition_name=partition_name,
                session=session,
                table_name=table_name,
            )
            partition_clauses.append(
                self._build_partition_clause(
                    quoted_partition=quoted_partition,
                    boundary_value=int(boundary_value),
                )
            )

        partition_str = ",\n".join(partition_clauses)

        ddl = f"""
            ALTER TABLE {quoted_table}
            PARTITION BY RANGE (partition_key) (
                {partition_str}
            )
        """
        mlrun.utils.logger.info(
            "Creating initial partitions",
            partitions_count=len(partition_list),
            table_name=table_name,
        )
        session.execute(sqlalchemy.text(ddl))
        session.commit()

    def _add_new_partitions(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        quoted_table: str,
        partition_list: list[tuple[str, int]],
        existing_partitions: list[tuple[str, int]],
    ) -> None:
        """
        Add future partitions whose upper bounds are above the current maximum.
        """
        max_existing_boundary = max(boundary for _, boundary in existing_partitions)

        new_partitions = [
            (partition_name, int(boundary_value))
            for partition_name, boundary_value in partition_list
            if int(boundary_value) > max_existing_boundary
        ]

        if not new_partitions:
            mlrun.utils.logger.debug(
                "No new partitions to add for table",
                table_name=table_name,
                max_existing_boundary=max_existing_boundary,
            )
            return

        partition_clauses = []
        for partition_name, boundary_value in new_partitions:
            quoted_partition, _ = self.get_quoted_partitioned_table_params(
                partition_name=partition_name,
                session=session,
                table_name=table_name,
            )
            partition_clauses.append(
                self._build_partition_clause(
                    quoted_partition=quoted_partition,
                    boundary_value=boundary_value,
                )
            )

        ddl = f"""
            ALTER TABLE {quoted_table}
            ADD PARTITION (
                {", ".join(partition_clauses)}
            )
        """
        mlrun.utils.logger.info(
            "Adding partitions",
            partitions_count=len(new_partitions),
            table_name=table_name,
            max_existing_boundary=max_existing_boundary,
        )
        session.execute(sqlalchemy.text(ddl))
        session.commit()

    @staticmethod
    def _build_partition_clause(
        quoted_partition: str,
        boundary_value: int,
    ) -> str:
        """
        Build a single PARTITION ... VALUES LESS THAN (...) clause for MySQL.
        """
        return f"PARTITION {quoted_partition} VALUES LESS THAN ({boundary_value})"


class PartitionBootstrapperPostgres(PartitionBootstrapper):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        """
        Create missing range partitions for a PostgreSQL partitioned table.

        Each new partition is anchored on the highest upper bound already present
        in the database rather than on its position in the freshly generated batch.
        This keeps partitions contiguous even when one or more intervals were
        missed (e.g. the periodic task did not run for a while): the first new
        partition simply extends from the newest existing bound and spans the gap,
        instead of re-emitting a ``MINVALUE`` lower bound that would overlap the
        original partition and abort the whole create/drop transaction.
        """
        partition_list = self._get_partition_list(
            table_name=table_name,
            partition_interval=partition_interval,
            partitions_count=partitions_count,
        )
        if not partition_list:
            return

        quoted_table = self._quote_table_name(
            session=session,
            table_name=table_name,
            sample_partition=partition_list[0][0],
        )

        existing_partitions = self._get_existing_partition_boundaries(
            session=session,
            table_name=table_name,
        )
        # Highest upper bound covered so far (existing + just-created). Each new
        # partition uses it as its lower bound, so partitions stay contiguous and
        # never overlap an existing one after a missed rotation. None means the
        # table has no partitions yet (genuine initial bootstrap).
        highest_covered_boundary = max(
            (boundary for _, boundary in existing_partitions),
            default=None,
        )

        mlrun.utils.logger.info(
            "Creating partitions (PostgreSQL)",
            requested_partitions=len(partition_list),
            existing_partitions=len(existing_partitions),
            highest_covered_boundary=highest_covered_boundary,
            table_name=table_name,
        )

        for partition_name, boundary_value in partition_list:
            upper_bound_value = int(boundary_value)

            # Skip anything already covered (by upper bound, not by name): this
            # keeps the run idempotent and avoids the range overlap that anchoring
            # on batch position — rather than on the DB — used to cause after a gap.
            if (
                highest_covered_boundary is not None
                and upper_bound_value <= highest_covered_boundary
            ):
                mlrun.utils.logger.debug(
                    "Partition range already covered, skipping creation",
                    table_name=table_name,
                    partition_name=partition_name,
                    upper_bound=upper_bound_value,
                    covered_up_to=highest_covered_boundary,
                )
                continue

            quoted_partition, _ = self.get_quoted_partitioned_table_params(
                partition_name=partition_name,
                session=session,
                table_name=table_name,
            )
            lower_bound = (
                "MINVALUE"
                if highest_covered_boundary is None
                else str(highest_covered_boundary)
            )
            upper_bound = str(upper_bound_value)
            ddl = f"""
                CREATE TABLE {quoted_partition}
                PARTITION OF {quoted_table}
                FOR VALUES FROM ({lower_bound}) TO ({upper_bound})
            """
            session.execute(sqlalchemy.text(ddl))
            highest_covered_boundary = upper_bound_value

        session.commit()

    def _get_existing_partition_boundaries(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
    ) -> list[tuple[str, int]]:
        """
        Read existing PostgreSQL child partitions and their integer upper bounds.

        :param session: SQLAlchemy session bound to the target database.
        :param table_name: Parent partitioned table name.
        :return: ``(partition_name, upper_bound)`` tuples. Rows whose upper bound
            cannot be parsed (e.g. a ``DEFAULT``/``MAXVALUE`` partition) are skipped.
        """
        sql = sqlalchemy.text(
            r"""
            SELECT c.relname AS partition_name,
                   (regexp_match(
                           pg_get_expr(c.relpartbound, c.oid),
                           'TO \((\d+)\)'
                    ))[1]::int AS upper_bound
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            JOIN pg_namespace n ON n.oid = p.relnamespace
            WHERE n.nspname = current_schema()
              AND p.relname = :table_name
              AND c.relkind = 'r'
            """
        )
        result = session.execute(sql, {"table_name": table_name})
        existing_partitions: list[tuple[str, int]] = []
        for partition_name, upper_bound in result:
            if upper_bound is None:
                continue
            existing_partitions.append((partition_name, int(upper_bound)))
        return existing_partitions


class PartitionBootstrapperSqlite(PartitionBootstrapper):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_interval: mlrun.common.schemas.partition_interval.PartitionInterval,
        partitions_count: int,
    ):
        """
        Log and skip partitioning for SQLite, which has no native partition support.
        """
        mlrun.utils.logger.info(
            "SQLite does not support partitioning natively, skipping bootstrap.",
            table_name=table_name,
        )
