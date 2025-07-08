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
import sqlalchemy

import mlrun

import framework.utils.db.utils
import framework.utils.singletons.db

mysql_engine = pytest_mock_resources.create_mysql_fixture()


@pytest.fixture
def alembic_engine(
    mysql_engine: sqlalchemy.engine.Engine,
) -> sqlalchemy.engine.Engine:
    os.environ["MLRUN_HTTPDB__DSN"] = mysql_engine.url.render_as_string(
        hide_password=False,
    )
    mlrun.mlconf.reload()
    framework.utils.singletons.db.initialize_db()
    return mysql_engine.execution_options(isolation_level="AUTOCOMMIT")


@pytest.fixture
def pmr_mysql_container(pytestconfig, pmr_mysql_config):
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_mysql_config,
        interval=1,
        retries=60,
    )


@pytest.fixture
def pmr_mysql_config():
    return pytest_mock_resources.MysqlConfig(
        image="mysql:8.0",
        port=3306,
        username="root",
        password="pass",
        root_database="mlrun",
    )


@pytest.fixture
def db_util(
    alembic_engine: sqlalchemy.engine.Engine,
) -> framework.utils.db.utils.DBUtil:
    util = framework.utils.db.utils.DBUtil()
    util.wait_for_db_liveness()
    return util
