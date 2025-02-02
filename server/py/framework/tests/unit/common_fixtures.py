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
import unittest.mock
from collections.abc import Generator
from tempfile import NamedTemporaryFile, TemporaryDirectory

import deepdiff
import fastapi
import httpx
import pytest
import pytest_asyncio
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
from mlrun.secrets import SecretsStore
from mlrun.utils import logger

import framework
import framework.utils.clients.iguazio
import framework.utils.projects.remotes.leader
import framework.utils.runtimes.nuclio
import framework.utils.singletons.db
import framework.utils.singletons.k8s
import framework.utils.singletons.project_member
from services.api.initial_data import init_data


class K8sSecretsMock(mlrun.common.secrets.InMemorySecretProvider):
    def __init__(self):
        super().__init__()
        self._is_running_in_k8s = True

    def reset_mock(self):
        # project -> secret_key -> secret_value
        self.project_secrets_map = {}
        # ref -> secret_key -> secret_value
        self.auth_secrets_map = {}
        # secret-name -> secret_key -> secret_value
        self.secrets_map = {}

    # cannot use a property since it's used as a method on the actual class
    def is_running_inside_kubernetes_cluster(self) -> bool:
        return self._is_running_in_k8s

    def set_is_running_in_k8s_cluster(self, value: bool):
        self._is_running_in_k8s = value

    def get_expected_env_variables_from_secrets(
        self, project, encode_key_names=True, include_internal=False, global_secret=None
    ):
        expected_env_from_secrets = {}

        if global_secret:
            for key in self.secrets_map.get(global_secret, {}):
                env_variable_name = (
                    SecretsStore.k8s_env_variable_name_for_secret(key)
                    if encode_key_names
                    else key
                )
                expected_env_from_secrets[env_variable_name] = {global_secret: key}

        secret_name = (
            framework.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
                project
            )
        )
        for key in self.project_secrets_map.get(project, {}):
            if key.startswith("mlrun.") and not include_internal:
                continue

            env_variable_name = (
                SecretsStore.k8s_env_variable_name_for_secret(key)
                if encode_key_names
                else key
            )
            expected_env_from_secrets[env_variable_name] = {secret_name: key}

        return expected_env_from_secrets

    def assert_project_secrets(self, project: str, secrets: dict):
        assert (
            deepdiff.DeepDiff(
                self.project_secrets_map[project],
                secrets,
                ignore_order=True,
            )
            == {}
        )

    def assert_auth_secret(self, secret_ref: str, username: str, access_key: str):
        assert (
            deepdiff.DeepDiff(
                self.auth_secrets_map[secret_ref],
                self._generate_auth_secret_data(username, access_key),
                ignore_order=True,
            )
            == {}
        )

    def store_secrets(self, secret_name, secrets: dict):
        secret_data = self.secrets_map.get(secret_name, {}).copy()

        # we don't care about encoding the value we want to store
        secret_data.update(secrets)
        self.secrets_map[secret_name] = secret_data

    def read_secret_data(self, secret_name, *args, **kwargs):
        return self.secrets_map.get(secret_name, {})

    def mock_functions(self, mocked_object, monkeypatch):
        mocked_function_names = [
            "is_running_inside_kubernetes_cluster",
            "get_project_secret_keys",
            "get_project_secret_data",
            "store_project_secrets",
            "delete_project_secrets",
            "store_auth_secret",
            "delete_auth_secret",
            "read_auth_secret",
            "get_secret_data",
            "store_secrets",
            "read_secret_data",
        ]

        for mocked_function_name in mocked_function_names:
            monkeypatch.setattr(
                mocked_object,
                mocked_function_name,
                getattr(self, mocked_function_name),
            )


class TestServiceBase:
    @pytest.fixture(scope="module")
    def app(self) -> fastapi.FastAPI:
        raise NotImplementedError(
            "Service application fixture should be implemented by the inheriting class"
        )

    @pytest.fixture(scope="module")
    def prefix(self):
        raise NotImplementedError(
            "Service API prefix fixture should be implemented by the inheriting class"
        )

    @pytest.fixture(autouse=True)
    def service_config_test(self):
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
    def db(self) -> typing.Iterator[sqlalchemy.orm.Session]:
        """
        This fixture initialize the db singleton (so it will be accessible using services.api.singletons.get_db()
        and generates a db session that can be used by the test
        """
        db_file = None
        try:
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
            # TODO: init data initializes the tables, we should remove this coupling with the API service code
            init_data(from_scratch=True)
            framework.utils.singletons.db.initialize_db()
            framework.utils.singletons.project_member.initialize_project_member()

            # we're also running client code in tests so set dbpath as well
            # note that setting this attribute triggers connection to the run db therefore must happen
            # after the initialization
            config.dbpath = dsn
            yield create_session()
        finally:
            if db_file:
                logger.info(f"Removing temp db file: {db_file.name}")
                db_file.close()

    def set_base_url_for_test_client(
        self,
        client: typing.Union[httpx.AsyncClient, TestClient],
        prefix: str,
    ):
        client.base_url = client.base_url.join(prefix)

    @pytest.fixture()
    def client(self, app, prefix) -> Generator:
        # skip partition management because it cannot be run on SQLite
        with unittest.mock.patch(
            "services.api.main.Service._start_periodic_partition_management",
            return_value=None,
        ):
            with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
                mlconf.httpdb.logs_path = log_dir
                mlconf.monitoring.runs.interval = 0
                mlconf.runtimes_cleanup_interval = 0
                mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"
                mlconf.httpdb.clusterization.chief.feature_gates.project_summaries = (
                    "false"
                )
                with TestClient(app) as test_client:
                    self.set_base_url_for_test_client(test_client, prefix)
                    yield test_client

    @pytest.fixture()
    def k8s_secrets_mock(self, monkeypatch) -> K8sSecretsMock:
        logger.info("Creating k8s secrets mock")
        k8s_secrets_mock = K8sSecretsMock()
        k8s_secrets_mock.mock_functions(
            framework.utils.singletons.k8s.get_k8s_helper(), monkeypatch
        )
        yield k8s_secrets_mock

    @pytest_asyncio.fixture()
    async def async_client(
        self, db, app, prefix
    ) -> typing.AsyncIterator[httpx.AsyncClient]:
        with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
            mlconf.httpdb.logs_path = log_dir
            mlconf.monitoring.runs.interval = 0
            mlconf.runtimes_cleanup_interval = 0
            mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

            async with httpx.AsyncClient(
                app=app, base_url="http://test"
            ) as async_client:
                self.set_base_url_for_test_client(async_client, prefix)
                yield async_client
