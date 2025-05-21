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
import logging
import os
import sys

import pytest
import pytest_mock_resources

mysql = pytest_mock_resources.create_mysql_fixture()

logger = logging.getLogger("pytest_mock_resources")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


@pytest.fixture(scope="function")
def alembic_engine(mysql):
    os.environ["MLRUN_HTTPDB__DSN"] = str(mysql.engine.url)
    import mlrun

    mlrun.config.config.reload()
    engine = mysql.engine

    engine = engine.execution_options(isolation_level="AUTOCOMMIT")
    return engine


@pytest.fixture(scope="function")
def pmr_mysql_container(pytestconfig, pmr_mysql_config):
    yield from pytest_mock_resources.get_container(
        pytestconfig=pytestconfig,
        config=pmr_mysql_config,
        interval=1,
        retries=60,
    )


@pytest.fixture(scope="function")
def pmr_mysql_config():
    return pytest_mock_resources.MysqlConfig(
        image="mysql:8.0",
        host="localhost",
        port=3306,
        username="root",
        password="pass",
        root_database="mlrun",
    )
