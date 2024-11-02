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
import datetime
import pathlib
import shutil
import typing
import unittest.mock
from collections.abc import Generator
from tempfile import NamedTemporaryFile, TemporaryDirectory
from http import HTTPStatus
import os

import requests
import v3io.dataplane.response
from aioresponses import aioresponses as aioresponses_

import deepdiff
import httpx
import pytest
import pytest_asyncio
import semver
import sqlalchemy.orm
from fastapi.testclient import TestClient

import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.db.factory
import mlrun.launcher.factory
import mlrun.runtimes.utils
import mlrun.utils.singleton
import services.api.crud
import services.api.launcher
import services.api.rundb.sqldb
import services.api.runtime_handlers.mpijob
import services.api.utils.clients.iguazio
import services.api.utils.projects.remotes.leader as project_leader
import services.api.utils.runtimes.nuclio
import services.api.utils.singletons.db
import services.api.utils.singletons.k8s
import services.api.utils.singletons.logs_dir
import services.api.utils.singletons.project_member
import services.api.utils.singletons.scheduler
from mlrun import mlconf
import mlrun_pipelines.utils
from mlrun.common.db.sql_session import _init_engine, create_session
from mlrun.config import config
from mlrun.secrets import SecretsStore
from mlrun.utils import logger
from services.api.initial_data import init_data
from services.api.main import API_PREFIX, BASE_VERSIONED_API_PREFIX, app

tests_root_directory = pathlib.Path(__file__).absolute().parent
assets_path = tests_root_directory.joinpath("assets")
results = tests_root_directory / "test_results"

os.environ["KFPMETA_OUT_DIR"] = f"{results}/kfp/"
os.environ["KFP_ARTIFACTS_DIR"] = f"{results}/kfp/"

rundb_path = f"{results}/rundb"
logs_path = f"{results}/logs"
out_path = f"{results}/out"
root_path = str(pathlib.Path(tests_root_directory).parent)
run_time_fmt = "%Y-%m-%dT%H:%M:%S.%fZ"


@pytest.fixture(autouse=True)
def api_config_test():
    # recreating the test results path on each test instead of running it on conftest since
    # it is not a threadsafe operation. if we'll run it on conftest it will be called multiple times
    # in parallel and may cause errors.
    shutil.rmtree(results, ignore_errors=True, onerror=None)
    pathlib.Path(f"{results}/kfp").mkdir(parents=True, exist_ok=True)

    services.api.utils.singletons.db.db = None
    services.api.utils.singletons.project_member.project_member = None
    services.api.utils.singletons.scheduler.scheduler = None
    services.api.utils.singletons.k8s._k8s = None
    services.api.utils.singletons.logs_dir.logs_dir = None

    os.environ["MLRUN_HTTPDB__DIRPATH"] = rundb_path
    os.environ["MLRUN_HTTPDB__LOGS_PATH"] = logs_path
    os.environ["MLRUN_HTTPDB__PROJECTS__PERIODIC_SYNC_INTERVAL"] = "0 seconds"
    os.environ["MLRUN_HTTPDB__PROJECTS__COUNTERS_CACHE_TTL"] = "0 seconds"
    os.environ["MLRUN_EXEC_CONFIG"] = ""
    mlrun.runtimes.utils.global_context.set(None)

    # reload config so that values overridden by tests won't pass to other tests
    mlrun.mlconf.reload()

    mlconf.nuclio_version = ""
    services.api.runtime_handlers.mpijob.cached_mpijob_crd_version = None

    mlrun.config._is_running_as_api = True
    services.api.utils.singletons.k8s.get_k8s_helper().running_inside_kubernetes_cluster = False

    # remove the store manager cache, so it won't pass between tests
    mlrun.datastore.store_manager._db = None
    mlrun.datastore.store_manager._stores = {}

    # no need to raise error when using nop_db
    mlrun.mlconf.httpdb.nop_db.raise_error = False
    # deploy status is mocked so no need to sleep
    mlrun.mlconf.httpdb.logs.nuclio.pull_deploy_status_default_interval = 0

    # we need to override the run db container manually because we run all unit tests in the same process in CI
    # so API is imported even when it's not needed
    rundb_factory = mlrun.db.factory.RunDBFactory()
    rundb_factory._rundb_container.override(services.api.rundb.sqldb.SQLRunDBContainer)

    # same for the launcher container
    launcher_factory = mlrun.launcher.factory.LauncherFactory()
    launcher_factory._launcher_container.override(
        services.api.launcher.ServerSideLauncherContainer
    )

    yield

    mlrun.config._is_running_as_api = None
    # remove singletons in case they were changed (we don't want changes to pass between tests)
    mlrun.utils.singleton.Singleton._instances = {}

    mlrun.runtimes.runtime_handler_instances_cache = {}

    # TODO: update this to "sidecar" once the default mode is changed
    mlrun.mlconf.log_collector.mode = "legacy"

    # revert change of default project after project creation
    mlrun.mlconf.default_project = "default"
    mlrun.projects.project.pipeline_context.set(None)

    # reset factory container overrides
    rundb_factory._rundb_container.reset_override()
    launcher_factory._launcher_container.reset_override()


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

    # forcing from scratch because we created an empty file for the db
    init_data(from_scratch=True)
    services.api.utils.singletons.db.initialize_db()
    services.api.utils.singletons.project_member.initialize_project_member()

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
        mlconf.httpdb.clusterization.chief.feature_gates.project_summaries = "false"
        with TestClient(app) as test_client:
            set_base_url_for_test_client(test_client)
            yield test_client


@pytest.fixture
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


@pytest_asyncio.fixture()
async def async_client(db) -> typing.AsyncIterator[httpx.AsyncClient]:
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.monitoring.runs.interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
            set_base_url_for_test_client(async_client)
            yield async_client


@pytest.fixture
def kfp_client_mock(monkeypatch) -> mlrun_pipelines.utils.kfp.Client:
    services.api.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
        return_value=True
    )
    kfp_client_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun_pipelines.utils.kfp, "Client", lambda *args, **kwargs: kfp_client_mock
    )
    mlrun.mlconf.kfp_url = "http://ml-pipeline.custom_namespace.svc.cluster.local:8888"
    return kfp_client_mock


@pytest.fixture()
def api_url() -> str:
    api_url = "http://iguazio-api-url:8080"
    mlrun.mlconf.iguazio_api_url = api_url
    return api_url


@pytest.fixture()
def iguazio_client(
    request: pytest.FixtureRequest,
) -> services.api.utils.clients.iguazio.Client:
    if request.param == "async":
        client = services.api.utils.clients.iguazio.AsyncClient()
    else:
        client = services.api.utils.clients.iguazio.Client()

    # force running init again so the configured api url will be used
    client.__init__()
    client._wait_for_job_completion_retry_interval = 0
    client._wait_for_project_terminal_state_retry_interval = 0

    # inject the request param into client, so we can use it in tests
    setattr(client, "mode", request.param)
    return client

# TODO: This fixture is duplicated with tests.common_fixtures.aioresponses_mock because we don't have a way to
#  share fixtures between client and server tests. Ideally we would use pytest --import-mode importlib to run the
#  server tests from the root directory, but that does not work ATM without changing the import path to include
#  server.py. See https://docs.pytest.org/en/stable/explanation/goodpractices.html#conventions-for-python-test-discovery
@pytest.fixture
def aioresponses_mock():
    with aioresponses_() as aior:
        # handy function to get how many times requests were made using this specific mock
        aior.called_times = lambda: len(list(aior.requests.values())[0])
        yield aior

def freeze(f, **kwargs):
    """
    Enables to override an attribute passed to a sub-function without the need to access the function directly
    :param f: the function we want to pass the attribute to
    :param kwargs: dictionary containing name(key) and value of the attributes to override.
    :return: wrapped function with overridden attributes
    """
    frozen = kwargs

    def wrapper(*args, **kwargs):
        kwargs.update(frozen)
        return f(*args, **kwargs)

    return wrapper

def run_now():
    return datetime.datetime.now().strftime(run_time_fmt)


def new_run(state, labels, uid=None, **kw):
    obj = {
        "metadata": {"name": "run-name", "labels": labels},
        "status": {"state": state, "start_time": run_now()},
    }
    if uid:
        obj["metadata"]["uid"] = uid
    obj.update(kw)
    return obj


class MockedK8sHelper:
    @pytest.fixture(autouse=True)
    def mock_k8s_helper(self):
        """
        This fixture mocks the k8s helper singleton for all tests in the class that inherit from this class.
        Example:
            class TestSomething(MockedK8sHelper):
                # Automatically uses the mocked k8s helper
                def test_something(self):
                    ...
        """
        _mocked_k8s_helper()


@pytest.fixture()
def mocked_k8s_helper():
    _mocked_k8s_helper()


def _mocked_k8s_helper():
    # We don't need to restore the original functions since the k8s cluster is never configured in unit tests
    services.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_keys = (
        unittest.mock.Mock(return_value=[])
    )
    services.api.utils.singletons.k8s.get_k8s_helper().v1api = unittest.mock.Mock()
    services.api.utils.singletons.k8s.get_k8s_helper().crdapi = unittest.mock.Mock()
    services.api.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
        return_value=True
    )

    config_map = unittest.mock.Mock()
    config_map.items = []
    services.api.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_config_map = unittest.mock.Mock(
        return_value=config_map
    )
    pods_list = unittest.mock.Mock()
    pods_list.items = []
    pods_list.metadata._continue = None
    services.api.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_pod = (
        unittest.mock.Mock(return_value=pods_list)
    )
    service_list = unittest.mock.Mock()
    service_list.items = []
    services.api.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_service = (
        unittest.mock.Mock(return_value=service_list)
    )
    custom_object_list = {"items": []}
    services.api.utils.singletons.k8s.get_k8s_helper().crdapi.list_namespaced_custom_object = unittest.mock.Mock(
        return_value=custom_object_list
    )
    secret_data = unittest.mock.Mock()
    secret_data.data = {}
    services.api.utils.singletons.k8s.get_k8s_helper().v1api.read_namespaced_secret = (
        unittest.mock.Mock(return_value=secret_data)
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
            services.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
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
                services.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    services.api.crud.secrets.SecretsClientType.service_accounts,
                    "default",
                )
            ] = default_service_account
        if allowed_service_accounts:
            secrets[
                services.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    services.api.crud.secrets.SecretsClientType.service_accounts,
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
        services.api.utils.singletons.k8s.get_k8s_helper(), monkeypatch
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
        services.api.crud.Projects().create_project(self._db_session, project)
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
    old_iguazio_client = services.api.utils.clients.iguazio.Client
    services.api.utils.clients.iguazio.Client = MockedProjectFollowerIguazioClient
    services.api.utils.singletons.project_member.initialize_project_member()
    iguazio_client = MockedProjectFollowerIguazioClient()
    iguazio_client._db_session = db
    iguazio_client._unversioned_client = unversioned_client

    yield iguazio_client

    services.api.utils.clients.iguazio.Client = old_iguazio_client


class MockSpecificCalls:
    def __init__(
        self,
        original_function: typing.Callable,
        call_indexes_to_mock: list[int],
        return_value: typing.Any,
    ):
        self.original_function = original_function
        self.call_indexes_to_mock = call_indexes_to_mock
        self.return_value = return_value

    calls = 0

    def mock_function(self, *args, **kwargs):
        self.calls += 1
        if self.calls not in self.call_indexes_to_mock:
            return self.original_function(*args, **kwargs)
        else:
            return self.return_value

# TODO: This fixture is duplicated with tests.common_fixtures.patch_file_forbidden because we don't have a way to
#  share fixtures between client and server tests. Ideally we would use pytest --import-mode importlib to run the
#  server tests from the root directory, but that does not work ATM without changing the import path to include
#  server.py. See https://docs.pytest.org/en/stable/explanation/goodpractices.html#conventions-for-python-test-discovery
@pytest.fixture
def patch_file_forbidden(monkeypatch):
    class MockV3ioObject:
        def get(self, *args, **kwargs):
            raise v3io.dataplane.response.HttpResponseError(
                "error", HTTPStatus.FORBIDDEN.value
            )

        def head(self, *args, **kwargs):
            raise v3io.dataplane.response.HttpResponseError(
                "error", HTTPStatus.FORBIDDEN.value
            )

    class MockV3ioClient:
        def __init__(self, *args, **kwargs):
            self.container = self

        def list(self, *args, **kwargs):
            raise RuntimeError("Permission denied")

        @property
        def object(self):
            return MockV3ioObject()

    mock_get = mock_failed_get_func(HTTPStatus.FORBIDDEN.value)

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "head", mock_get)
    monkeypatch.setattr(v3io.dataplane, "Client", MockV3ioClient)

@pytest.fixture
def patch_file_not_found(monkeypatch):
    class MockV3ioObject:
        def get(self, *args, **kwargs):
            raise v3io.dataplane.response.HttpResponseError(
                "error", HTTPStatus.NOT_FOUND.value
            )

        def head(self, *args, **kwargs):
            raise v3io.dataplane.response.HttpResponseError(
                "error", HTTPStatus.NOT_FOUND.value
            )

    class MockV3ioClient:
        def __init__(self, *args, **kwargs):
            self.container = self

        def list(self, *args, **kwargs):
            raise FileNotFoundError

        @property
        def object(self):
            return MockV3ioObject()

    mock_get = mock_failed_get_func(HTTPStatus.NOT_FOUND.value)

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "head", mock_get)
    monkeypatch.setattr(v3io.dataplane, "Client", MockV3ioClient)

def mock_failed_get_func(status_code: int):
    def mock_get(*args, **kwargs):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = status_code
        mock_response.raise_for_status = unittest.mock.Mock(
            side_effect=requests.HTTPError("Error", response=mock_response)
        )
        return mock_response

    return mock_get