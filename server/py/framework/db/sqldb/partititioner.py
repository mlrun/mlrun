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
            return super().__new__(MySQLRangePartitioner)
        if dialect.startswith(mlrun.common.db.dialects.Dialects.POSTGRES):
            return super().__new__(PostgresRangePartitioner)
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


class MySQLRangePartitioner(RangePartitioner):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_expression: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        prep = sqlalchemy.sql.compiler.IdentifierPreparer(session.get_bind().dialect)
        q_table = prep.quote(table_name)
        q_part = prep.quote(first_partition_name)
        session.execute(
            sqlalchemy.text(
                f"""ALTER TABLE {q_table}
                    PARTITION BY RANGE ({partition_expression})
                    (PARTITION {q_part} VALUES LESS THAN ({int(first_partition_upper_bound)}))"""
            )
        )
        session.commit()


class PostgresRangePartitioner(RangePartitioner):
    def bootstrap(
        self,
        session: sqlalchemy.orm.Session,
        table_name: str,
        partition_expression: str,
        first_partition_name: str,
        first_partition_upper_bound: str,
    ):
        prep = sqlalchemy.sql.compiler.IdentifierPreparer(session.get_bind().dialect)
        q_table = prep.quote(table_name)
        q_part = prep.quote(first_partition_name)
        session.execute(
            sqlalchemy.text(
                f"ALTER TABLE {q_table} PARTITION BY RANGE ({partition_expression})"
            )
        )
        session.execute(
            sqlalchemy.text(
                f"""CREATE TABLE {q_part}
                     PARTITION OF {q_table}
                     FOR VALUES FROM (MINVALUE) TO ({int(first_partition_upper_bound)})"""
            )
        )
        session.commit()
