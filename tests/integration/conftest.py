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
import pytest_mock_resources as pmr
import sqlalchemy
from _pytest.config import Config
from sqlalchemy import inspect

import mlrun
from mlrun.common.db.dialects import Dialects

logger = mlrun.utils.create_test_logger(name="test.integration.conftest")


@pytest.fixture(scope="session")
def pmr_mysql_container(
    pytestconfig: Config, pmr_mysql_config: pytest_mock_resources.MysqlConfig
):
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_mysql_config,
        interval=1,
        retries=60,
    )


@pytest.fixture(scope="session")
def pmr_mysql_config() -> pytest_mock_resources.MysqlConfig:
    return pytest_mock_resources.MysqlConfig(
        image="mysql:8.0",
        port=3306,
        username="root",
        password="pass",
        root_database="mlrun",
    )


_mysql_engine = pmr.create_mysql_fixture(
    scope="session",
)


def _wipe_database(engine):
    """Truncate all user tables & reset sequences."""
    insp = inspect(engine)
    with engine.begin() as conn:
        if engine.dialect.name.startswith(Dialects.MYSQL):
            conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 0"))
            for t in insp.get_table_names():
                conn.execute(sqlalchemy.text(f"DROP TABLE `{t}`"))
            conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 1"))
        else:
            raise ValueError(f"Unsupported database dialect: {engine.dialect.name}")


@pytest.fixture
def db_engine(
    request: pytest.FixtureRequest, _mysql_engine
) -> sqlalchemy.engine.Engine:
    db_type = os.getenv("MLRUN_TEST_DB", "mysql").lower()
    logger.info("Starting database engine", db_type=db_type)
    yield _mysql_engine
    logger.info("Wiping database", db_type=db_type)
    _wipe_database(_mysql_engine)
