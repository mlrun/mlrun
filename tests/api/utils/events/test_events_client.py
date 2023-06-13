import unittest.mock

import fastapi.testclient
import pytest
import semver
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.events.events_factory
import mlrun.common.schemas
import tests.api.conftest


class TestEventClient:
    @pytest.mark.parametrize(
        "iguazio_version",
        [
            "3.5.4",
            "3.5.3",
            None,
        ],
    )
    def test_create_project_auth_secret(
        self,
        monkeypatch,
        db: sqlalchemy.orm.Session,
        client: fastapi.testclient.TestClient,
        k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
        iguazio_version: str,
    ):
        self._initialize_and_mock_client(monkeypatch, iguazio_version)

        username = "some-username"
        access_key = "some-access-key"
        mlrun.api.crud.Secrets().store_auth_secret(
            mlrun.common.schemas.AuthSecretData(
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                username=username,
                access_key=access_key,
            )
        )
        self.__assert_client_was_called(iguazio_version)

    @pytest.mark.parametrize(
        "iguazio_version",
        [
            "3.5.4",
            "3.5.3",
            None,
        ],
    )
    def test_create_project_secret(
        self,
        monkeypatch,
        db: sqlalchemy.orm.Session,
        client: fastapi.testclient.TestClient,
        k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
        iguazio_version: str,
    ):
        self._initialize_and_mock_client(monkeypatch, iguazio_version)

        project = "project-name"
        valid_secret_key = "valid-key"
        valid_secret_value = "some-value-5"
        provider = mlrun.common.schemas.SecretProviderName.kubernetes
        key_map_secret_key = (
            mlrun.api.crud.Secrets().generate_client_key_map_project_secret_key(
                mlrun.api.crud.SecretsClientType.schedules
            )
        )
        mlrun.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={valid_secret_key: valid_secret_value}
            ),
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )

        self.__assert_client_was_called(iguazio_version)

    def __assert_client_was_called(self, iguazio_version: str):
        self.client.emit.assert_called_once()
        if iguazio_version:
            if iguazio_version >= semver.VersionInfo.parse("3.5.4-b1"):
                # if equal or greater than 3.5.4-b1, then we expect the event to exist in the events.yaml of the system
                # and therefor we send the minimized event (without the description)
                assert not self.client.emit.call_args[0][0].description
            else:
                assert self.client.emit.call_args[0][0].description
        else:
            assert self.client.emit.call_args[0][0] is None

    def _initialize_and_mock_client(self, monkeypatch, iguazio_version: str):
        mlrun.mlconf.events.mode = mlrun.common.schemas.EventsModes.enabled.value
        self._initialize_client(iguazio_version)
        self.client.emit = unittest.mock.MagicMock()
        monkeypatch.setattr(
            mlrun.api.utils.events.events_factory.EventsFactory,
            "get_events_client",
            lambda *args, **kwargs: self.client,
        )

    def _initialize_client(self, version: str = None):
        mlrun.mlconf.igz_version = version
        self.client = (
            mlrun.api.utils.events.events_factory.EventsFactory.get_events_client()
        )
