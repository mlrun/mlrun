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
from collections.abc import Generator, Iterator

import pytest
import pytest_mock_resources
import sqlalchemy
import sqlalchemy.event
from _pytest.config import Config

import mlrun

import framework.utils.db.utils
import framework.utils.singletons.db

mysql_engine = pytest_mock_resources.create_mysql_fixture(scope="session")


# TODO: Remove me once we squash all old alembic revisions that create FKs
# that reference non-unique columns.
def _set_mysql_session_variables(dbapi_connection, connection_record):
    """
    Event listener to set MySQL session variables on every new connection.
    This ensures FK constraints can reference non-unique columns (MySQL 8.4+).
    """
    cursor = dbapi_connection.cursor()
    try:
        # MySQL 8.4+ requires unique keys for FK references by default.
        # Disable this restriction at session level for each connection.
        cursor.execute("SET SESSION restrict_fk_on_non_standard_key = OFF")
    finally:
        cursor.close()


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
def alembic_engine(
    mysql_engine: sqlalchemy.engine.Engine,
) -> sqlalchemy.engine.Engine:
    db_url = mysql_engine.url
    db_name = db_url.database
    admin_url = db_url.set(database=None)
    admin_engine = sqlalchemy.create_engine(admin_url)

    with admin_engine.connect() as conn:
        conn.execute(sqlalchemy.text(f"DROP DATABASE IF EXISTS `{db_name}`"))
        conn.execute(sqlalchemy.text(f"CREATE DATABASE `{db_name}`"))

    # Register event listener to set MySQL session variables on every connection.
    # This ensures FK constraints can reference non-unique columns (MySQL 8.4+).
    sqlalchemy.event.listen(mysql_engine, "connect", _set_mysql_session_variables)

    # Force any existing connections to be closed so new ones get the event listener
    mysql_engine.dispose()

    os.environ["MLRUN_HTTPDB__DSN"] = db_url.render_as_string(hide_password=False)
    mlrun.mlconf.reload()
    framework.utils.singletons.db.initialize_db()

    return mysql_engine.execution_options(isolation_level="AUTOCOMMIT")


@pytest.fixture(scope="session")
def pmr_mysql_container(
    pytestconfig: Config,
    pmr_mysql_config: pytest_mock_resources.MysqlConfig,
) -> Iterator[None]:
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_mysql_config,
        interval=1,
        retries=60,
    )


@pytest.fixture(scope="session")
def db_util(
    alembic_engine: sqlalchemy.engine.Engine,
) -> Generator[framework.utils.db.utils.DBUtil, None, None]:
    util = framework.utils.db.utils.DBUtil()
    util.wait_for_db_liveness()
    yield util
