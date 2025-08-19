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

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.compiler

import mlrun.common.db.dialects


class RangePartitioner:
    def __new__(cls, dialect: str):
        if dialect.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
            return super().__new__(RangePartitionerMySQL)
        if dialect.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL):
            return super().__new__(RangePartitionerPostgres)
        raise ValueError(dialect)

    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_expression: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        raise NotImplementedError

    def get_quoted_partitioned_table_params(
        self,
        first_partition_name: str,
        session: sqlalchemy.orm.Session,
        table_name: str,
    ) -> tuple[str, str]:
        preparer = sqlalchemy.sql.compiler.IdentifierPreparer(
            session.get_bind().dialect
        )
        quoted_table_name = preparer.quote(table_name)
        quoted_partition_name = preparer.quote(first_partition_name)
        return quoted_partition_name, quoted_table_name


class RangePartitionerMySQL(RangePartitioner):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_expression: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        quoted_partition_name, quoted_table_name = (
            self.get_quoted_partitioned_table_params(
                first_partition_name=first_partition_name,
                session=session,
                table_name=table_name,
            )
        )
        session.execute(
            sqlalchemy.text(
                f"""ALTER TABLE {quoted_table_name}
                    PARTITION BY RANGE ({partition_expression})
                    (PARTITION {quoted_partition_name} VALUES LESS THAN ({int(first_partition_upper_bound)}))"""
            )
        )
        session.commit()


class RangePartitionerPostgres(RangePartitioner):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_expression: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        quoted_partition_name, quoted_table_name = (
            self.get_quoted_partitioned_table_params(
                first_partition_name=first_partition_name,
                session=session,
                table_name=table_name,
            )
        )
        session.execute(
            sqlalchemy.text(
                f"ALTER TABLE {quoted_table_name} PARTITION BY RANGE ({partition_expression})"
            )
        )
        session.execute(
            sqlalchemy.text(
                f"""CREATE TABLE {quoted_partition_name}
                     PARTITION OF {quoted_table_name}
                     FOR VALUES FROM (MINVALUE) TO ({int(first_partition_upper_bound)})"""
            )
        )
        session.commit()
