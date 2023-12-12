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
import os
from http import HTTPStatus
from unittest.mock import Mock

import requests

import mlrun
import mlrun.errors
from mlrun.platforms import add_or_refresh_credentials


def test_add_or_refresh_credentials_iguazio_2_8_success(monkeypatch):
    username = "username"
    password = "password"
    control_session = "control_session"
    api_url = "https://dashboard.default-tenant.app.hedingber-28-1.iguazio-cd2.com"
    env = os.environ
    env["V3IO_USERNAME"] = username
    env["V3IO_PASSWORD"] = password

    def mock_get(*args, **kwargs):
        not_found_response_mock = Mock()
        not_found_response_mock.ok = False
        not_found_response_mock.status_code = HTTPStatus.NOT_FOUND.value
        return not_found_response_mock

    def mock_session(*args, **kwargs):
        session_mock = Mock()

        def _mock_successful_session_creation(*args, **kwargs):
            assert session_mock.auth == (username, password)
            successful_response_mock = Mock()
            successful_response_mock.ok = True
            successful_response_mock.json.return_value = {
                "data": {"id": control_session}
            }
            return successful_response_mock

        session_mock.post = _mock_successful_session_creation
        return session_mock

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "Session", mock_session)

    result_username, result_control_session, _ = add_or_refresh_credentials(api_url)
    assert username == result_username
    assert control_session == result_control_session


def test_add_or_refresh_credentials_iguazio_2_10_success(monkeypatch):
    username = "username"
    access_key = "access_key"
    api_url = "https://dashboard.default-tenant.app.hedingber-210-1.iguazio-cd2.com"
    env = os.environ
    env["V3IO_USERNAME"] = username
    env["V3IO_ACCESS_KEY"] = access_key

    def mock_get(*args, **kwargs):
        ok_response_mock = Mock()
        ok_response_mock.ok = True
        return ok_response_mock

    monkeypatch.setattr(requests, "get", mock_get)

    result_username, result_access_key, _ = add_or_refresh_credentials(api_url)
    assert username == result_username
    assert access_key == result_access_key


def test_add_or_refresh_credentials_kubernetes_svc_url_success(monkeypatch):
    access_key = "access_key"
    api_url = "http://mlrun-api:8080"
    env = os.environ
    env["V3IO_ACCESS_KEY"] = access_key

    _, _, result_access_key = add_or_refresh_credentials(api_url)
    assert access_key == result_access_key


def test_is_iguazio_session_cookie():
    assert (
        mlrun.platforms.is_iguazio_session_cookie(
            "j%3A%7B%22sid%22%3A%20%22946b0749-5c40-4837-a4ac-341d295bfaf7%22%7D"
        )
        is True
    )
    assert mlrun.platforms.is_iguazio_session_cookie("dummy") is False
