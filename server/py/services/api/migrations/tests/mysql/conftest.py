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
from _pytest.config import Config
from pytest_mock_resources import MysqlConfig

import mlrun

import framework.utils.singletons.db
from framework.utils.db.utils import DBUtil

mysql = pytest_mock_resources.create_mysql_fixture()


@pytest.fixture(autouse=True)
def patched_dsn(mysql):
    os.environ["MLRUN_HTTPDB__DSN"] = str(mysql.engine.url)
    mlrun.mlconf.reload()


@pytest.fixture
def alembic_engine(mysql):
    engine = mysql.engine
    framework.utils.singletons.db.initialize_db()
    engine = engine.execution_options(isolation_level="AUTOCOMMIT")
    return engine


@pytest.fixture
def pmr_mysql_container(pytestconfig: Config, pmr_mysql_config: MysqlConfig):
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_mysql_config,
        interval=1,
        retries=60,
    )

@pytest.fixture
def db_util() -> DBUtil:
    util = DBUtil()
    return util


@pytest.fixture
def pmr_mysql_config():
    return pytest_mock_resources.MysqlConfig(
        image="mysql:8.0",
        host="localhost",
        port=3306,
        username="root",
        password="pass",
        root_database="mlrun",
    )
