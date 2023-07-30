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
import typing
import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generator

import deepdiff
import httpx
import kfp
import pytest
from fastapi.testclient import TestClient

import mlrun.api.launcher
import mlrun.api.rundb.sqldb
import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.runtimes.nuclio
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.api.utils.singletons.logs_dir
import mlrun.api.utils.singletons.project_member
import mlrun.api.utils.singletons.scheduler
import mlrun.common.schemas
import mlrun.db.factory
import mlrun.launcher.factory
from mlrun import mlconf
from mlrun.api.initial_data import init_data
from mlrun.api.main import BASE_VERSIONED_API_PREFIX, app
from mlrun.common.db.sql_session import _init_engine, create_session
from mlrun.config import config
from mlrun.secrets import SecretsStore
from mlrun.utils import logger


@pytest.fixture(autouse=True)
def api_config_test():
    mlrun.api.utils.singletons.db.db = None
    mlrun.api.utils.singletons.project_member.project_member = None
    mlrun.api.utils.singletons.scheduler.scheduler = None
    mlrun.api.utils.singletons.k8s._k8s = None
    mlrun.api.utils.singletons.logs_dir.logs_dir = None

    mlrun.api.utils.runtimes.nuclio.cached_nuclio_version = None

    mlrun.config._is_running_as_api = True

    # we need to override the run db container manually because we run all unit tests in the same process in CI
    # so API is imported even when it's not needed
    rundb_factory = mlrun.db.factory.RunDBFactory()
    rundb_factory._rundb_container.override(mlrun.api.rundb.sqldb.SQLRunDBContainer)

    # same for the launcher container
    launcher_factory = mlrun.launcher.factory.LauncherFactory()
    launcher_factory._launcher_container.override(
        mlrun.api.launcher.ServerSideLauncherContainer
    )

    yield

    mlrun.config._is_running_as_api = None

    # reset factory container overrides
    rundb_factory._rundb_container.reset_override()
    launcher_factory._launcher_container.reset_override()


@pytest.fixture()
def db() -> Generator:
    """
    This fixture initialize the db singleton (so it will be accessible using mlrun.api.singletons.get_db()
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
    mlrun.api.utils.singletons.db.initialize_db()
    mlrun.api.utils.singletons.project_member.initialize_project_member()

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
        mlconf.runs_monitoring_interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        with TestClient(app) as test_client:
            set_base_url_for_test_client(test_client)
            yield test_client


@pytest.fixture()
@pytest.mark.asyncio
async def async_client(db) -> Generator:
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.runs_monitoring_interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
            set_base_url_for_test_client(async_client)
            yield async_client


class K8sSecretsMock:
    def __init__(self):
        self._is_running_in_k8s = True
        self.reset_mock()

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

    @staticmethod
    def get_auth_secret_name(username: str, access_key: str) -> str:
        return f"secret-ref-{username}-{access_key}"

    def store_auth_secret(
        self, username: str, access_key: str, namespace=""
    ) -> (str, mlrun.common.schemas.SecretEventActions):
        secret_ref = self.get_auth_secret_name(username, access_key)
        self.auth_secrets_map.setdefault(secret_ref, {}).update(
            self._generate_auth_secret_data(username, access_key)
        )
        return secret_ref, mlrun.common.schemas.SecretEventActions.created

    @staticmethod
    def _generate_auth_secret_data(username: str, access_key: str):
        return {
            mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                "username"
            ): username,
            mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                "access_key"
            ): access_key,
        }

    def delete_auth_secret(self, secret_ref: str, namespace=""):
        del self.auth_secrets_map[secret_ref]

    def read_auth_secret(self, secret_name, namespace="", raise_on_not_found=False):
        secret = self.auth_secrets_map.get(secret_name)
        if not secret:
            if raise_on_not_found:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Secret '{secret_name}' was not found in auth secrets map"
                )

            return None, None
        username = secret[
            mlrun.common.schemas.AuthSecretData.get_field_secret_key("username")
        ]
        access_key = secret[
            mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key")
        ]
        return username, access_key

    def store_project_secrets(
        self, project, secrets, namespace=""
    ) -> (str, mlrun.common.schemas.SecretEventActions):
        self.project_secrets_map.setdefault(project, {}).update(secrets)
        secret_name = project
        return secret_name, mlrun.common.schemas.SecretEventActions.created

    def delete_project_secrets(self, project, secrets, namespace=""):
        if not secrets:
            self.project_secrets_map.pop(project, None)
        else:
            for key in secrets:
                self.project_secrets_map.get(project, {}).pop(key, None)
        return "", True

    def get_project_secret_keys(self, project, namespace="", filter_internal=False):
        secret_keys = list(self.project_secrets_map.get(project, {}).keys())
        if filter_internal:
            secret_keys = list(
                filter(lambda key: not key.startswith("mlrun."), secret_keys)
            )
        return secret_keys

    def get_project_secret_data(self, project, secret_keys=None, namespace=""):
        secrets_data = self.project_secrets_map.get(project, {})
        return {
            key: value
            for key, value in secrets_data.items()
            if (secret_keys and key in secret_keys) or not secret_keys
        }

    def store_secret(self, secret_name, secrets: dict):
        self.secrets_map[secret_name] = secrets

    def get_secret_data(self, secret_name, namespace=""):
        return self.secrets_map[secret_name]

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
            mlrun.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
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
                mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    mlrun.api.crud.secrets.SecretsClientType.service_accounts, "default"
                )
            ] = default_service_account
        if allowed_service_accounts:
            secrets[
                mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    mlrun.api.crud.secrets.SecretsClientType.service_accounts, "allowed"
                )
            ] = ",".join(allowed_service_accounts)
        self.store_project_secrets(project, secrets)


@pytest.fixture()
def k8s_secrets_mock(monkeypatch, client: TestClient) -> K8sSecretsMock:
    logger.info("Creating k8s secrets mock")
    k8s_secrets_mock = K8sSecretsMock()

    mocked_function_names = [
        "is_running_inside_kubernetes_cluster",
        "get_project_secret_keys",
        "get_project_secret_data",
        "store_project_secrets",
        "delete_project_secrets",
        "get_auth_secret_name",
        "store_auth_secret",
        "delete_auth_secret",
        "read_auth_secret",
        "get_secret_data",
    ]

    for mocked_function_name in mocked_function_names:
        monkeypatch.setattr(
            mlrun.api.utils.singletons.k8s.get_k8s_helper(),
            mocked_function_name,
            getattr(k8s_secrets_mock, mocked_function_name),
        )

    yield k8s_secrets_mock


@pytest.fixture
def kfp_client_mock(monkeypatch) -> kfp.Client:
    mlrun.api.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
        return_value=True
    )
    kfp_client_mock = unittest.mock.Mock()
    monkeypatch.setattr(kfp, "Client", lambda *args, **kwargs: kfp_client_mock)
    mlrun.mlconf.kfp_url = "http://ml-pipeline.custom_namespace.svc.cluster.local:8888"
    return kfp_client_mock


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://iguazio-api-url:8080"
    mlrun.config.config.iguazio_api_url = api_url
    return api_url


@pytest.fixture()
async def iguazio_client(
    api_url: str,
    request: pytest.FixtureRequest,
) -> mlrun.api.utils.clients.iguazio.Client:
    if request.param == "async":
        client = mlrun.api.utils.clients.iguazio.AsyncClient()
    else:
        client = mlrun.api.utils.clients.iguazio.Client()

    # force running init again so the configured api url will be used
    client.__init__()
    client._wait_for_job_completion_retry_interval = 0
    client._wait_for_project_terminal_state_retry_interval = 0

    # inject the request param into client, so we can use it in tests
    setattr(client, "mode", request.param)
    return client
