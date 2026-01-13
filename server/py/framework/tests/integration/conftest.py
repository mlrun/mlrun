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
import typing

import pytest
import pytest_mock_resources
import sqlalchemy
import sqlalchemy.orm

import mlrun.common.db.dialects
import mlrun.utils

import framework.db.sqldb.models
import framework.utils.singletons.db

logger = mlrun.utils.create_test_logger()


# Determine which backend is under test
TEST_DB = os.getenv("MLRUN_TEST_DB", mlrun.common.db.dialects.Dialects.MYSQL)

_mysql_engine = pytest_mock_resources.create_mysql_fixture(
    scope="session",
)

_postgres_engine = pytest_mock_resources.create_postgres_fixture(
    scope="session",
)


@pytest.fixture(scope="session")
def pmr_mysql_config() -> pytest_mock_resources.MysqlConfig:
    return pytest_mock_resources.MysqlConfig(
        image=os.getenv("MLRUN_MYSQL_IMAGE", "gcr.io/iguazio/mlrun-mysql:8.4"),
        port=3306,
        username="root",
        password="pass",
        root_database="mlrun",
    )


@pytest.fixture(scope="session")
def pmr_postgres_config() -> pytest_mock_resources.PostgresConfig:
    return pytest_mock_resources.PostgresConfig(
        image=os.getenv("MLRUN_POSTGRES_IMAGE", "gcr.io/iguazio/postgres:17"),
        port=5432,
        username="root",
        password="pass",
        root_database="mlrun",
        drivername="postgresql+psycopg2",
    )


def _wipe_database(
    engine: sqlalchemy.engine.Engine,
) -> None:
    """Truncate all user tables & reset sequences."""
    insp = sqlalchemy.inspect(engine)
    with engine.begin() as conn:
        if engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL):
            tables = insp.get_table_names(schema="public")
            if tables:
                conn.execute(
                    sqlalchemy.text(
                        "DROP TABLE " + ", ".join(f'"{t}"' for t in tables) + " CASCADE"
                    )
                )
        elif engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
            conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 0"))
            for t in insp.get_table_names():
                conn.execute(sqlalchemy.text(f"DROP TABLE `{t}`"))
            conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 1"))
        elif engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.SQLITE):
            tables = insp.get_table_names()
            if tables:
                for table in tables:
                    conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table}`"))
        else:
            raise ValueError(f"Unsupported database dialect: {engine.dialect.name}")


@pytest.fixture
def db_engine(
    request: pytest.FixtureRequest,
) -> typing.Generator[sqlalchemy.engine.Engine, None, None]:
    db_type = os.getenv("MLRUN_TEST_DB", "mysql").lower()
    logger.info("Starting database engine", db_type=db_type)

    engine: sqlalchemy.engine.Engine = request.getfixturevalue(
        "_postgres_engine" if db_type == "postgres" else "_mysql_engine"
    )

    logger.info("Started database engine", db_type=db_type)
    os.environ["MLRUN_HTTPDB__DSN"] = engine.url.render_as_string(hide_password=False)
    mlrun.mlconf.reload()
    logger.info("Wiping database", db_type=db_type)
    _wipe_database(engine)
    framework.utils.singletons.db.initialize_db()
    yield engine
