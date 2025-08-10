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
import sqlalchemy.engine
from _pytest.config import Config

import mlrun

import framework.utils.db.utils
import framework.utils.singletons.db

pytest.importorskip(
    "psycopg2",
    reason="psycopg2 not installed",
)

postgres_engine = pytest_mock_resources.create_postgres_fixture(scope="session")


@pytest.fixture(scope="session")
def pmr_postgres_config() -> pytest_mock_resources.PostgresConfig:
    return pytest_mock_resources.PostgresConfig(
        image="postgres:17",
        port=5432,
        username="root",
        password="pass",
        root_database="mlrun",
        drivername="postgresql+psycopg2",
    )


@pytest.fixture(scope="session")
def alembic_engine(
    postgres_engine: sqlalchemy.engine.Engine,
) -> sqlalchemy.engine.Engine:
    db_url = postgres_engine.url
    db_name = db_url.database

    postgres_engine.dispose()

    admin_url = db_url.set(database="postgres")
    admin_engine = sqlalchemy.create_engine(admin_url)
    raw_conn = admin_engine.raw_connection()
    try:
        raw_conn.connection.set_session(autocommit=True)
        with raw_conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            cur.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        raw_conn.close()
        admin_engine.dispose()

    os.environ["MLRUN_HTTPDB__DSN"] = postgres_engine.url.render_as_string(
        hide_password=False
    )
    mlrun.mlconf.reload()
    framework.utils.singletons.db.initialize_db()
    return postgres_engine.execution_options(isolation_level="AUTOCOMMIT")


@pytest.fixture(scope="session")
def pmr_postgres_container(
    pytestconfig: Config,
    pmr_postgres_config: pytest_mock_resources.PostgresConfig,
) -> Iterator[None]:
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_postgres_config,
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
