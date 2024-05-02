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
#
import datetime
import typing
import unittest.mock
from collections.abc import Generator
from tempfile import NamedTemporaryFile, TemporaryDirectory

import deepdiff
import httpx
import kfp
import pytest
import semver
import sqlalchemy.orm
from fastapi.testclient import TestClient

import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.db.factory
import mlrun.launcher.factory
import mlrun.utils.singleton
import server.api.crud
import server.api.launcher
import server.api.rundb.sqldb
import server.api.runtime_handlers.mpijob
import server.api.utils.clients.iguazio
import server.api.utils.projects.remotes.leader as project_leader
import server.api.utils.runtimes.nuclio
import server.api.utils.singletons.db
import server.api.utils.singletons.k8s
import server.api.utils.singletons.logs_dir
import server.api.utils.singletons.project_member
import server.api.utils.singletons.scheduler
from mlrun import mlconf
from mlrun.common.db.sql_session import _init_engine, create_session
from mlrun.config import config
from mlrun.secrets import SecretsStore
from mlrun.utils import logger
from server.api.initial_data import init_data
from server.api.main import API_PREFIX, BASE_VERSIONED_API_PREFIX, app


@pytest.fixture(autouse=True)
def api_config_test():
    server.api.utils.singletons.db.db = None
    server.api.utils.singletons.project_member.project_member = None
    server.api.utils.singletons.scheduler.scheduler = None
    server.api.utils.singletons.k8s._k8s = None
    server.api.utils.singletons.logs_dir.logs_dir = None

    server.api.utils.runtimes.nuclio.cached_nuclio_version = None
    server.api.runtime_handlers.mpijob.cached_mpijob_crd_version = None

    mlrun.config._is_running_as_api = True
    server.api.utils.singletons.k8s.get_k8s_helper().running_inside_kubernetes_cluster = False

    # we need to override the run db container manually because we run all unit tests in the same process in CI
    # so API is imported even when it's not needed
    rundb_factory = mlrun.db.factory.RunDBFactory()
    rundb_factory._rundb_container.override(server.api.rundb.sqldb.SQLRunDBContainer)

    # same for the launcher container
    launcher_factory = mlrun.launcher.factory.LauncherFactory()
    launcher_factory._launcher_container.override(
        server.api.launcher.ServerSideLauncherContainer
    )

    yield

    mlrun.config._is_running_as_api = None

    # reset factory container overrides
    rundb_factory._rundb_container.reset_override()
    launcher_factory._launcher_container.reset_override()


@pytest.fixture()
def db() -> Generator:
    """
    This fixture initialize the db singleton (so it will be accessible using server.api.singletons.get_db()
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

    # forcing from scratch because we created an empty file for the db
    init_data(from_scratch=True)
    server.api.utils.singletons.db.initialize_db()
    server.api.utils.singletons.project_member.initialize_project_member()

    # we're also running client code in tests so set dbpath as well
    # note that setting this attribute triggers connection to the run db therefore must happen after the initialization
    config.dbpath = dsn
    yield create_session()
    logger.info(f"Removing temp db file: {db_file.name}")
    db_file.close()


def set_base_url_for_test_client(
    client: typing.Union[httpx.AsyncClient, TestClient],
    prefix: str = BASE_VERSIONED_API_PREFIX,
):
    client.base_url = client.base_url.join(prefix)


@pytest.fixture()
def client(db) -> Generator:
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.monitoring.runs.interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        with TestClient(app) as test_client:
            set_base_url_for_test_client(test_client)
            yield test_client


@pytest.fixture()
def unversioned_client(db) -> Generator:
    """
    unversioned_client is a test client that doesn't have the version prefix in the url.
    When using this client, the version prefix must be added to the url manually.
    This is useful when tests use several endpoints that are not under the same version prefix.
    """
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.monitoring.runs.interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        with TestClient(app) as unversioned_test_client:
            set_base_url_for_test_client(unversioned_test_client, API_PREFIX)
            yield unversioned_test_client


@pytest.fixture()
@pytest.mark.asyncio
async def async_client(db) -> Generator:
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.monitoring.runs.interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
            set_base_url_for_test_client(async_client)
            yield async_client


@pytest.fixture
def kfp_client_mock(monkeypatch) -> kfp.Client:
    server.api.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
        return_value=True
    )
    kfp_client_mock = unittest.mock.Mock()
    monkeypatch.setattr(kfp, "Client", lambda *args, **kwargs: kfp_client_mock)
    mlrun.mlconf.kfp_url = "http://ml-pipeline.custom_namespace.svc.cluster.local:8888"
    return kfp_client_mock


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://iguazio-api-url:8080"
    mlrun.mlconf.iguazio_api_url = api_url
    return api_url


@pytest.fixture()
async def iguazio_client(
    api_url: str,
    request: pytest.FixtureRequest,
) -> server.api.utils.clients.iguazio.Client:
    if request.param == "async":
        client = server.api.utils.clients.iguazio.AsyncClient()
    else:
        client = server.api.utils.clients.iguazio.Client()

    # force running init again so the configured api url will be used
    client.__init__()
    client._wait_for_job_completion_retry_interval = 0
    client._wait_for_project_terminal_state_retry_interval = 0

    # inject the request param into client, so we can use it in tests
    setattr(client, "mode", request.param)
    return client


class MockedK8sHelper:
    @pytest.fixture(autouse=True)
    def mock_k8s_helper(self, db: sqlalchemy.orm.Session, client: TestClient):
        # We need the client fixture (which needs the db one) in order to be able to mock k8s stuff
        # We don't need to restore the original functions since the k8s cluster is never configured in unit tests
        server.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_keys = (
            unittest.mock.Mock(return_value=[])
        )
        server.api.utils.singletons.k8s.get_k8s_helper().v1api = unittest.mock.Mock()
        server.api.utils.singletons.k8s.get_k8s_helper().crdapi = unittest.mock.Mock()
        server.api.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
            return_value=True
        )


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
            server.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
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

    def set_service_account_keys(
        self, project, default_service_account, allowed_service_accounts
    ):
        secrets = {}
        if default_service_account:
            secrets[
                server.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    server.api.crud.secrets.SecretsClientType.service_accounts,
                    "default",
                )
            ] = default_service_account
        if allowed_service_accounts:
            secrets[
                server.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    server.api.crud.secrets.SecretsClientType.service_accounts,
                    "allowed",
                )
            ] = ",".join(allowed_service_accounts)
        self.store_project_secrets(project, secrets)

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
        ]

        for mocked_function_name in mocked_function_names:
            monkeypatch.setattr(
                mocked_object,
                mocked_function_name,
                getattr(self, mocked_function_name),
            )


@pytest.fixture()
def k8s_secrets_mock(monkeypatch) -> K8sSecretsMock:
    logger.info("Creating k8s secrets mock")
    k8s_secrets_mock = K8sSecretsMock()
    k8s_secrets_mock.mock_functions(
        server.api.utils.singletons.k8s.get_k8s_helper(), monkeypatch
    )
    yield k8s_secrets_mock


class MockedProjectFollowerIguazioClient(
    project_leader.Member, metaclass=mlrun.utils.singleton.AbstractSingleton
):
    def __init__(self):
        self._db_session = None
        self._unversioned_client = None

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        wait_for_completion: bool = True,
    ) -> bool:
        server.api.crud.Projects().create_project(self._db_session, project)
        return False

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
    ):
        pass

    def delete_project(
        self,
        session: str,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        api_version = "v2"
        igz_version = mlrun.mlconf.get_parsed_igz_version()
        if igz_version and igz_version < semver.VersionInfo.parse("3.5.5"):
            api_version = "v1"

        self._unversioned_client.delete(
            f"{api_version}/projects/{name}",
            headers={
                mlrun.common.schemas.HeaderNames.projects_role: mlrun.mlconf.httpdb.projects.leader,
                mlrun.common.schemas.HeaderNames.deletion_strategy: deletion_strategy,
            },
        )

        # Mock waiting for completion in iguazio (return False to indicate 'not running in background')
        return False

    def list_projects(
        self,
        session: str,
        updated_after: typing.Optional[datetime.datetime] = None,
    ) -> tuple[list[mlrun.common.schemas.Project], typing.Optional[datetime.datetime]]:
        return [], None

    def get_project(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.Project:
        pass

    def format_as_leader_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.IguazioProject:
        pass

    def get_project_owner(
        self,
        session: str,
        name: str,
    ) -> mlrun.common.schemas.ProjectOwner:
        pass


@pytest.fixture()
def mock_project_follower_iguazio_client(
    db: sqlalchemy.orm.Session, unversioned_client: TestClient
):
    """
    This fixture mocks the project leader iguazio client.
    """
    mlrun.mlconf.httpdb.projects.leader = "iguazio"
    mlrun.mlconf.httpdb.projects.iguazio_access_key = "access_key"
    old_iguazio_client = server.api.utils.clients.iguazio.Client
    server.api.utils.clients.iguazio.Client = MockedProjectFollowerIguazioClient
    server.api.utils.singletons.project_member.initialize_project_member()
    iguazio_client = MockedProjectFollowerIguazioClient()
    iguazio_client._db_session = db
    iguazio_client._unversioned_client = unversioned_client

    yield iguazio_client

    server.api.utils.clients.iguazio.Client = old_iguazio_client
