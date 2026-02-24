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
import pathlib
import unittest.mock
from collections.abc import Generator, Iterator
from datetime import datetime
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import fastapi
import kfp_server_api
import pytest
import semver
import sqlalchemy.orm
from fastapi.testclient import TestClient
from kfp_server_api.models.api_experiment import ApiExperiment

import mlrun
import mlrun.common.schemas
import mlrun.launcher.factory
import mlrun.utils
import mlrun.utils.singleton
import mlrun_pipelines.client
import mlrun_pipelines.utils

import framework.utils.clients.iguazio.v3
import framework.utils.clients.iguazio.v4
import framework.utils.projects.remotes.leader
import framework.utils.singletons.k8s
import services.api.crud
import services.api.daemon
import services.api.launcher
import services.api.runtime_handlers.mpijob
import services.api.utils.singletons.logs_dir
import services.api.utils.singletons.scheduler
from framework.tests.unit.common_fixtures import K8sSecretsMock, TestServiceBase
from services.api.daemon import daemon

tests_root_directory = pathlib.Path(__file__).absolute().parent
assets_path = tests_root_directory.joinpath("assets")


class TestAPIBase(TestServiceBase):
    @pytest.fixture(scope="module")
    def app(self) -> Iterator[fastapi.FastAPI]:
        mlrun.mlconf.services.service_name = "api"
        mlrun.mlconf.services.hydra.services = ""
        yield services.api.daemon.app()

    @pytest.fixture(scope="module")
    def prefix(self):
        yield daemon.service.base_versioned_service_prefix

    # TODO: Move this to common fixtures similar to framework.tests.unit.common_fixtures.client
    @pytest.fixture
    def unversioned_client(self, db, app) -> Generator:
        """
        unversioned_client is a test client that doesn't have the version prefix in the url.
        When using this client, the version prefix must be added to the url manually.
        This is useful when tests use several endpoints that are not under the same version prefix.
        """
        with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
            mlrun.mlconf.httpdb.logs_path = log_dir
            mlrun.mlconf.monitoring.runs.interval = 0
            mlrun.mlconf.runtimes_cleanup_interval = 0
            mlrun.mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

            with TestClient(app) as unversioned_test_client:
                self.set_base_url_for_test_client(
                    unversioned_test_client, daemon.service.service_prefix
                )
                yield unversioned_test_client


# TODO: This is a hack to allow sharing fixtures between services in non-root directives because pytest behavior
#  changes with respect to the directive in which the test is running from. To use the common fixtures we need to use
#  pytest plugins but it is not allowed in non-root directive which means the fixture must apply on all tests
#  including client side. The correct way to solve this is using TestAPIBase class like in alerts service unit tests
#  but it is a big refactor for this PR
test_api_base = TestAPIBase()
service_config_test = test_api_base.service_config_test
app = test_api_base.app
prefix = test_api_base.prefix
db = test_api_base.db
set_base_url_for_test_client = test_api_base.set_base_url_for_test_client
client = test_api_base.client
unversioned_client = test_api_base.unversioned_client
async_client = test_api_base.async_client


@pytest.fixture(autouse=True)
def api_config_test(service_config_test):
    framework.utils.singletons.project_member.project_member = None
    services.api.utils.singletons.scheduler.scheduler = None
    services.api.utils.singletons.logs_dir.logs_dir = None

    services.api.runtime_handlers.mpijob.cached_mpijob_crd_version = None

    # we need to override the containers manually because we run all unit tests in
    # the same process in CI so services are imported even when they are not needed
    launcher_factory = mlrun.launcher.factory.LauncherFactory()
    launcher_factory._launcher_container.override(
        services.api.launcher.ServerSideLauncherContainer
    )
    service_container = framework.service.ServiceContainer()
    service_container.override(services.api.daemon.APIServiceContainer)

    yield
    launcher_factory._launcher_container.reset_override()
    service_container.reset_override()


@pytest.fixture
def kfp_client_mock(monkeypatch):
    framework.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = mock.Mock(
        return_value=True
    )
    client_klass = mlrun_pipelines.client.Client

    monkeypatch.setattr("kubernetes.config.load_incluster_config", lambda: None)
    monkeypatch.setattr(client_klass, "_determine_server_major_version", lambda self: 2)
    mock_experiment_api = mock.Mock()
    monkeypatch.setattr(
        kfp_server_api.api.experiment_service_api,
        "ExperimentServiceApi",
        mock.Mock(return_value=mock_experiment_api),
    )
    mock_experiment_api.list_experiment = mock.Mock(
        return_value=SimpleNamespace(
            experiments=[
                ApiExperiment(name="some-project"),
                ApiExperiment(name="another"),
            ]
        )
    )
    mock_experiment_api.api_client = mock.Mock()
    mock_experiment_api.api_client.call_api = mock.Mock()

    # Mock the KFP Run API; tests can stub methods on this as needed
    mock_run_api = mock.Mock()
    mock_run_api.create_run = mock.Mock()
    # It’s common that list_runs is used in pipeline listing; leave it mockable
    mock_run_api.list_runs = mock.Mock(return_value=SimpleNamespace(runs=[]))
    monkeypatch.setattr(
        kfp_server_api.api.run_service_api,
        "RunServiceApi",
        mock.Mock(return_value=mock_run_api),
    )

    # Build a real mlrun_pipelines client that will use our mocked APIs
    kfp_client = mlrun_pipelines.client.Client(logger=mock.Mock())
    # Point mlrun to a fake in-cluster KFP URL (not actually contacted due to mocks)
    mlrun.mlconf.kfp_url = "http://ml-pipeline.custom_namespace.svc.cluster.local:8888"

    # When code calls utils.get_client(...), hand back our prepared client
    monkeypatch.setattr(
        mlrun_pipelines.utils,
        "get_client",
        lambda *unused_args, **unused_kwargs: kfp_client,
    )

    return kfp_client


@pytest.fixture()
def api_url() -> str:
    api_url = "http://iguazio-api-url:8080"
    mlrun.mlconf.iguazio_api_url = api_url
    return api_url


@pytest.fixture()
def iguazio_client(
    request: pytest.FixtureRequest,
):
    """
    A parameterized fixture to return either an IG3 or IG4 client (sync or async)
    based on request parameters.

    Usage:
        @pytest.mark.parametrize(
            "iguazio_client",
            [("v3", "async"), ("v4", "sync")],
            indirect=True
        )
    """
    version, mode = request.param

    if version == "v3":
        module = framework.utils.clients.iguazio.v3
        client_cls = module.Client if mode == "sync" else module.AsyncClient
        client = client_cls()
    elif version == "v4":
        module = framework.utils.clients.iguazio.v4
        client_cls = module.Client if mode == "sync" else module.AsyncClient

        # PATCH iguazio.Client before instantiation
        with unittest.mock.patch(
            "framework.utils.clients.iguazio.v4.iguazio.Client"
        ) as mock_iguazio_cls:
            mock_instance = unittest.mock.MagicMock()
            mock_iguazio_cls.return_value = mock_instance

            # Now when Client.__init__ runs, self._client is assigned to mock_instance
            client = client_cls()
    else:
        raise ValueError(f"Unsupported client version: {version}")

    client._wait_for_job_completion_retry_interval = 0

    # inject the request param into client, so we can use it in tests
    setattr(client, "mode", request.param)
    return client


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
    framework.utils.singletons.k8s.get_k8s_helper().get_project_secret_keys = (
        unittest.mock.Mock(return_value=[])
    )
    framework.utils.singletons.k8s.get_k8s_helper().v1api = unittest.mock.Mock()
    framework.utils.singletons.k8s.get_k8s_helper().crdapi = unittest.mock.Mock()
    framework.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
        return_value=True
    )

    config_map = unittest.mock.Mock()
    config_map.items = []
    framework.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_config_map = (
        unittest.mock.Mock(return_value=config_map)
    )
    pods_list = unittest.mock.Mock()
    pods_list.items = []
    pods_list.metadata._continue = None
    framework.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_pod = (
        unittest.mock.Mock(return_value=pods_list)
    )
    service_list = unittest.mock.Mock()
    service_list.items = []
    framework.utils.singletons.k8s.get_k8s_helper().v1api.list_namespaced_service = (
        unittest.mock.Mock(return_value=service_list)
    )
    custom_object_list = {"items": []}
    framework.utils.singletons.k8s.get_k8s_helper().crdapi.list_namespaced_custom_object = unittest.mock.Mock(
        return_value=custom_object_list
    )
    secret_data = unittest.mock.Mock()
    secret_data.data = {}
    framework.utils.singletons.k8s.get_k8s_helper().v1api.read_namespaced_secret = (
        unittest.mock.Mock(return_value=secret_data)
    )


class APIK8sSecretsMock(K8sSecretsMock):
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


@pytest.fixture()
def k8s_secrets_mock(monkeypatch) -> APIK8sSecretsMock:
    mlrun.utils.logger.info("Creating k8s secrets mock")
    k8s_secrets_mock = APIK8sSecretsMock()
    k8s_secrets_mock.mock_functions(
        framework.utils.singletons.k8s.get_k8s_helper(), monkeypatch
    )
    yield k8s_secrets_mock


class MockedProjectFollowerIguazioClient(
    framework.utils.projects.remotes.leader.Member,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self):
        self._db_session = None
        self._unversioned_client = None

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        services.api.crud.Projects().create_project(self._db_session, project)
        return False

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        pass

    def delete_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
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
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        updated_after: datetime | None = None,
    ) -> tuple[list[mlrun.common.schemas.Project], datetime | None]:
        return [], None

    def get_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
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
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
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
    old_iguazio_client = framework.utils.clients.iguazio.v3.Client
    framework.utils.clients.iguazio.v3.Client = MockedProjectFollowerIguazioClient
    framework.utils.singletons.project_member.initialize_project_member()
    iguazio_client = MockedProjectFollowerIguazioClient()
    iguazio_client._db_session = db
    iguazio_client._unversioned_client = unversioned_client

    yield iguazio_client

    framework.utils.clients.iguazio.v3.Client = old_iguazio_client
