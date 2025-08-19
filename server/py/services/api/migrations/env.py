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
import contextlib
import logging.config
import typing

import alembic
import alembic.runtime.migration
import sqlalchemy
import sqlalchemy.dialects
import sqlalchemy.exc
import sqlalchemy.pool
import sqlalchemy.sql.type_api

import mlrun.utils

import framework.db.sqldb.lock_killer
import framework.db.sqldb.models
import framework.db.sqldb.sql_types

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = alembic.context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
logging.config.fileConfig(config.config_file_name, disable_existing_loggers=False)

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
    context: alembic.runtime.migration.MigrationContext,
    inspected_column: sqlalchemy.Column[typing.Any],
    metadata_column: sqlalchemy.Column[typing.Any],
    inspected_type: sqlalchemy.sql.type_api.TypeEngine[typing.Any],
    metadata_type: sqlalchemy.sql.type_api.TypeEngine[typing.Any],
) -> typing.Optional[bool]:
    """Custom compare_type that:
    1. checks mysql.VARCHAR→Utf8BinText by length+collation (utf8mb3_bin≈utf8_bin),
    2. suppresses VARCHAR→Uuid/UuidType only if length matches,
    3. flags DATETIME/TIMESTAMP→DateTime/MicroSecondDateTime only on fsp mismatch,
    4. flags PostgreSQL TIMESTAMP precision mismatches,
    otherwise defers to Alembic default."""
    if isinstance(inspected_type, sqlalchemy.dialects.mysql.VARCHAR):
        # suppress VARCHAR→Uuid/UuidType only if lengths are equal
        if isinstance(
            metadata_column.type,
            (sqlalchemy.Uuid, framework.db.sqldb.sql_types.UuidType),
        ):
            inspected_len = getattr(inspected_type, "length", None)
            meta_len = getattr(metadata_column.type, "length", None)
            return False if inspected_len == meta_len else True

        # handle Utf8BinText by collation + length
        coll = (inspected_type.collation or "").lower()
        if coll in ("utf8mb3_bin", "utf8_bin"):
            if isinstance(
                metadata_column.type, framework.db.sqldb.sql_types.Utf8BinText
            ):
                dialect = context.dialect
                meta_impl = metadata_column.type.load_dialect_impl(dialect)
                if getattr(inspected_type, "length", None) == getattr(
                    meta_impl, "length", None
                ):
                    return False
                return True

    # DATETIME/TIMESTAMP → DateTime/MicroSecondDateTime (MySQL)
    if isinstance(
        inspected_type,
        (sqlalchemy.dialects.mysql.DATETIME, sqlalchemy.dialects.mysql.TIMESTAMP),
    ) and isinstance(
        metadata_column.type,
        (
            framework.db.sqldb.sql_types.DateTime,
            framework.db.sqldb.sql_types.MicroSecondDateTime,
        ),
    ):
        if getattr(inspected_type, "fsp", None) == metadata_column.type.precision:
            return False
        return True

    # TIMESTAMP precision mismatches (PostgreSQL)
    if isinstance(
        inspected_type, sqlalchemy.dialects.postgresql.TIMESTAMP
    ) and isinstance(
        metadata_column.type,
        (
            framework.db.sqldb.sql_types.DateTime,
            framework.db.sqldb.sql_types.MicroSecondDateTime,
        ),
    ):
        if getattr(inspected_type, "precision", None) == metadata_column.type.precision:
            return False
        return True

    return None


@contextlib.contextmanager
def _get_connection():
    connection_or_engine = alembic.context.config.attributes.get("connection")

    if connection_or_engine is None:
        engine = sqlalchemy.engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=sqlalchemy.pool.NullPool,
        )
        with engine.connect() as conn:
            yield conn
        return

    # Figure out what Alembic passed in `config.attributes["connection"]`
    #
    # None         – developer runs `alembic upgrade` directly.
    #                Build a one-off Engine from alembic.ini and connect.
    # Engine       – regular runtime upgrades (API start, /operations/migrations).
    #                We must create and close a short-lived Connection ourselves.
    # Connection   – first-time bootstrap or unit-tests that already started
    #                  `engine.begin()`.  Caller owns the transaction; just yield it.
    if isinstance(connection_or_engine, sqlalchemy.engine.Engine):
        with connection_or_engine.connect() as conn:
            yield conn
        return

    if isinstance(connection_or_engine, sqlalchemy.engine.Connection):
        yield connection_or_engine
        return

    raise TypeError(f"Unsupported connection type: {type(connection_or_engine)!r}")


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


def _kill_locks(connection: sqlalchemy.engine.Connection):
    try:
        framework.db.sqldb.lock_killer.LockKiller(connection).kill_locks()
    except NotImplementedError:
        mlrun.utils.logger.info(
            "Lock killing not implemented",
            dialect=connection.dialect.name,
        )


def run_migrations_online():
    """
    Run migrations in *online* mode.
    """
    connectable = alembic.context.config.attributes.get("connection")

    if connectable is None:
        connectable = sqlalchemy.engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=sqlalchemy.pool.NullPool,
        )

    # Engine  → normal upgrades (API start, /operations/migrations, etc.):
    #            open a temp conn, run, close.
    # Connection → first-time bootstrap or tests (caller opened TX); reuse as-is.
    # (None → plain `alembic upgrade` CLI, handled earlier.)
    if isinstance(connectable, sqlalchemy.engine.Connection):
        connection = connectable
        close_conn = False
    elif isinstance(connectable, sqlalchemy.engine.Engine):
        connection = connectable.connect()
        close_conn = True
    else:
        raise TypeError(
            "Expected sqlalchemy.engine.Connection or sqlalchemy.engine.Engine"
        )

    try:
        _kill_locks(connection)
        alembic.context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=compare_type,
        )
        with alembic.context.begin_transaction():
            alembic.context.run_migrations()

        if connection.in_transaction():
            connection.commit()
    finally:
        if close_conn:
            connection.close()


if alembic.context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
