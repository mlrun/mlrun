# Copyright 2023 Iguazio
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
import typing
from collections.abc import Generator
from tempfile import NamedTemporaryFile, TemporaryDirectory

import httpx
import pytest
import sqlalchemy.orm
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.engine import Engine

import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.db.factory
import mlrun.launcher.factory
import mlrun.runtimes.utils
import mlrun.utils.singleton
from mlrun import mlconf
from mlrun.common.db.sql_session import _init_engine, create_session
from mlrun.config import config
from mlrun.utils import logger

import framework
import framework.utils.clients.iguazio
import framework.utils.projects.remotes.leader
import framework.utils.runtimes.nuclio
import framework.utils.singletons.db
import framework.utils.singletons.k8s
from services.api.initial_data import init_data


@pytest.fixture(autouse=True)
def service_config_test():
    framework.utils.singletons.db.db = None
    framework.utils.singletons.k8s._k8s = None

    mlconf.nuclio_version = ""

    mlrun.config._is_running_as_api = True
    framework.utils.singletons.k8s.get_k8s_helper().running_inside_kubernetes_cluster = False

    # we need to override the run db container manually because we run all unit tests in the same process in CI
    # so API is imported even when it's not needed
    rundb_factory = mlrun.db.factory.RunDBFactory()
    rundb_factory._rundb_container.override(framework.rundb.sqldb.SQLRunDBContainer)

    yield

    mlrun.config._is_running_as_api = None

    # reset factory container overrides
    rundb_factory._rundb_container.reset_override()


@pytest.fixture()
def db() -> typing.Iterator[sqlalchemy.orm.Session]:
    """
    This fixture initialize the db singleton (so it will be accessible using services.api.singletons.get_db()
    and generates a db session that can be used by the test
    """
    db_file = NamedTemporaryFile(suffix="-mlrun.db")
    logger.info(f"Created temp db file: {db_file.name}")
    config.httpdb.db_type = "sqldb"
    dsn = f"sqlite:///{db_file.name}?check_same_thread=false"
    config.httpdb.dsn = dsn
    mlrun.config._is_running_as_api = True

    # TODO: make it simpler - doesn't make sense to call 3 different functions to initialize the db
    # we need to force re-init the engine cause otherwise it is cached between tests
    _init_engine(dsn=config.httpdb.dsn)

    # SQLite foreign keys constraint must be enabled manually to allow cascade deletions on DB level
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # forcing from scratch because we created an empty file for the db
    init_data(from_scratch=True)
    framework.utils.singletons.db.initialize_db()
    framework.utils.singletons.project_member.initialize_project_member()

    # we're also running client code in tests so set dbpath as well
    # note that setting this attribute triggers connection to the run db therefore must happen after the initialization
    config.dbpath = dsn
    yield create_session()
    logger.info(f"Removing temp db file: {db_file.name}")
    db_file.close()


def set_base_url_for_test_client(
    client: typing.Union[httpx.AsyncClient, TestClient],
    prefix: str,
):
    client.base_url = client.base_url.join(prefix)


@pytest.fixture()
def client(db, app, prefix) -> Generator:
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.monitoring.runs.interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"
        mlconf.httpdb.clusterization.chief.feature_gates.project_summaries = "false"
        with TestClient(app) as test_client:
            set_base_url_for_test_client(test_client, prefix)
            yield test_client
