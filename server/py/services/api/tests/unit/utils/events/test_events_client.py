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
import unittest.mock
from typing import Optional

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.schemas

import services.api.crud
import services.api.tests.unit.conftest
import services.api.utils.events.events_factory


class TestEventClient:
    @pytest.mark.parametrize(
        "iguazio_version",
        [
            "3.5.4",
            "3.5.3",
            None,
        ],
    )
    def test_emit_project_auth_secret_event(
        self,
        monkeypatch,
        db: sqlalchemy.orm.Session,
        client: fastapi.testclient.TestClient,
        k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
        iguazio_version: str,
    ):
        # since auth secrets are internal we don't emit events when they are created/updated/deleted,
        # so we just emit from the client for testing purposes
        self._initialize_and_mock_client(monkeypatch, iguazio_version)

        username = "some-username"
        events_client = (
            services.api.utils.events.events_factory.EventsFactory().get_events_client()
        )
        event = events_client.generate_auth_secret_event(
            username=username,
            secret_name="auth_secret_name",
            action=mlrun.common.schemas.AuthSecretEventActions.created,
        )
        events_client.emit(event)
        self._assert_client_was_called(iguazio_version)

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
        k8s_secrets_mock: services.api.tests.unit.conftest.APIK8sSecretsMock,
        iguazio_version: str,
    ):
        self._initialize_and_mock_client(monkeypatch, iguazio_version)

        project = "project-name"
        valid_secret_key = "valid-key"
        valid_secret_value = "some-value-5"
        provider = mlrun.common.schemas.SecretProviderName.kubernetes
        key_map_secret_key = (
            services.api.crud.Secrets().generate_client_key_map_project_secret_key(
                services.api.crud.SecretsClientType.schedules
            )
        )
        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=provider, secrets={valid_secret_key: valid_secret_value}
            ),
            allow_internal_secrets=True,
            key_map_secret_key=key_map_secret_key,
        )

        self._assert_client_was_called(iguazio_version)

    def _initialize_and_mock_client(self, monkeypatch, iguazio_version: str):
        mlrun.mlconf.events.mode = mlrun.common.schemas.EventsModes.enabled.value
        self._initialize_client(iguazio_version)
        self.client.emit = unittest.mock.MagicMock()
        monkeypatch.setattr(
            services.api.utils.events.events_factory.EventsFactory,
            "get_events_client",
            lambda *args, **kwargs: self.client,
        )

    def _initialize_client(self, version: Optional[str] = None):
        mlrun.mlconf.igz_version = version
        self.client = (
            services.api.utils.events.events_factory.EventsFactory.get_events_client()
        )

    def _assert_client_was_called(self, iguazio_version: str):
        self.client.emit.assert_called_once()
        if iguazio_version:
            assert self.client.emit.call_args[0][0].description
        else:
            assert self.client.emit.call_args[0][0] is None
