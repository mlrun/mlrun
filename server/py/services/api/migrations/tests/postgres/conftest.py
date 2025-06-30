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

import pytest
import pytest_mock_resources

import mlrun

import framework.utils.singletons.db

# Abort import of this file unless the Postgres extra is available
pytest.importorskip(
    "pytest_mock_resources.postgres",
    reason="pytest-mock-resources[postgres] not installed",
)
postgres = pytest_mock_resources.create_postgres_fixture()


@pytest.fixture
def alembic_engine(postgres):
    os.environ["MLRUN_HTTPDB__DSN"] = str(postgres.engine.url)
    mlrun.mlconf.reload()
    engine = postgres.engine
    framework.utils.singletons.db.initialize_db()

    engine = engine.execution_options(isolation_level="AUTOCOMMIT")
    return engine


@pytest.fixture
def pmr_postgres_config():
    return pytest_mock_resources.PostgresConfig(
        image="postgres:17",
        host="localhost",
        port=5432,
        username="root",
        password="pass",
        root_database="mlrun",
    )


@pytest.fixture
def pmr_postgres_container(pytestconfig, pmr_postgres_config):
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_postgres_config,
        interval=1,
        retries=60,
    )
