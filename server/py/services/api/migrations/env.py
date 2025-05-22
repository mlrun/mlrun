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
from logging.config import fileConfig
from typing import Any, Optional

import alembic
import sqlalchemy
import sqlalchemy.dialects.mysql
import sqlalchemy.dialects.postgresql
import sqlalchemy.exc
import sqlalchemy.pool
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Column, Uuid
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.sql.type_api import TypeEngine

import mlrun.utils
from mlrun.db.sql_types import DateTime, MicroSecondDateTime, Utf8BinText

import framework.db.sqldb.models

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = alembic.context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = framework.db.sqldb.models.Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# this will overwrite the ini-file sqlalchemy.url path
# with the path given in the mlconf
config.set_main_option("sqlalchemy.url", mlrun.mlconf.httpdb.dsn)


# This function was added as part of the migration to SQLAlchemy 2.0 and is intended
# to suppress redundant alembic migrations
def compare_type(
    context: MigrationContext,
    inspected_column: Column[Any],
    metadata_column: Column[Any],
    inspected_type: TypeEngine[Any],
    metadata_type: TypeEngine[Any],
) -> Optional[bool]:
    """Custom compare_type that:
    1. checks mysql.VARCHAR→Utf8BinText by length+collation (utf8mb3_bin≈utf8_bin),
    2. suppresses VARCHAR→Uuid when collation is binary,
    3. flags DATETIME/TIMESTAMP→DateTime/MicroSecondDateTime only on fsp mismatch,
    4. flags PostgreSQL TIMESTAMP precision mismatches,
    otherwise defers to Alembic default."""
    if isinstance(inspected_type, mysql.VARCHAR):
        coll = (inspected_type.collation or "").lower()
        if coll in ("utf8mb3_bin", "utf8_bin"):
            if isinstance(metadata_column.type, Utf8BinText):
                dialect = context.dialect
                meta_impl = metadata_column.type.load_dialect_impl(dialect)
                if getattr(inspected_type, "length", None) == getattr(
                    meta_impl, "length", None
                ):
                    return False
                else:
                    return True
            # Uuid case
            if isinstance(metadata_column.type, Uuid):
                return False

    if isinstance(inspected_type, (mysql.DATETIME, mysql.TIMESTAMP)) and isinstance(
        metadata_column.type, (DateTime, MicroSecondDateTime)
    ):
        if getattr(inspected_type, "fsp", None) == metadata_column.type.precision:
            return False
        return True

    if isinstance(inspected_type, postgresql.TIMESTAMP) and isinstance(
        metadata_column.type, (DateTime, MicroSecondDateTime)
    ):
        if getattr(inspected_type, "precision", None) == metadata_column.type.precision:
            return False
        return True

    return None


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    alembic.context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=compare_type,
    )

    with alembic.context.begin_transaction():
        alembic.context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = alembic.context.config.attributes.get("connection", None)

    if connectable is None:
        connect_args = {}
        connectable = sqlalchemy.engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=sqlalchemy.pool.NullPool,
            connect_args=connect_args,
        )

    with connectable.connect() as connection:
        if connection.dialect.name == "mysql":
            # This query retrieves information about database connections that have acquired
            # locks on objects in the 'mlrun' schema,
            # excluding the current connection and the 'alembic_version' table.
            # This is order to find and kill connections that might be blocking the migration.
            connection_ids = connection.execute(
                sqlalchemy.sql.text(
                    """SELECT
                    t.PROCESSLIST_ID,
                    t.PROCESSLIST_USER,
                    t.PROCESSLIST_HOST,
                    GROUP_CONCAT(DISTINCT ml.OBJECT_NAME ORDER BY ml.OBJECT_NAME SEPARATOR ', ') AS locked_objects
                FROM
                    performance_schema.metadata_locks AS ml
                INNER JOIN
                    performance_schema.threads AS t
                    ON ml.OWNER_THREAD_ID = t.THREAD_ID
                WHERE
                    t.PROCESSLIST_ID <> CONNECTION_ID()
                    AND ml.OBJECT_SCHEMA = 'mlrun'
                    AND ml.OBJECT_NAME != 'alembic_version'
                    AND ml.LOCK_STATUS = 'GRANTED'
                GROUP BY
                    t.PROCESSLIST_ID,
                    t.PROCESSLIST_USER,
                    t.PROCESSLIST_HOST
                ORDER BY
                    t.PROCESSLIST_ID;
                """
                )
            ).fetchall()
            for connection_id, user, host, locked_objects in connection_ids:
                mlrun.utils.logger.warning(
                    "Killing DB connection with acquired lock.",
                    connection_id=connection_id,
                    user=user,
                    host=host,
                    locked_objects=locked_objects,
                    db="mlrun",
                )
                try:
                    connection.execute(sqlalchemy.sql.text(f"KILL {connection_id};"))
                except sqlalchemy.exc.OperationalError as exc:
                    if "Unknown thread id" in str(exc):
                        mlrun.utils.logger.warning(
                            "DB connection already closed.",
                            connection_id=connection_id,
                            user=user,
                            host=host,
                            db="mlrun",
                        )
                    else:
                        raise exc

        alembic.context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=compare_type,
        )

        with alembic.context.begin_transaction():
            alembic.context.run_migrations()


if alembic.context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
