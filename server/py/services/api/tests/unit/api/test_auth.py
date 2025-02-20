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
import http

import aioresponses
import fastapi.testclient
import sqlalchemy.orm
import starlette.datastructures

import mlrun.common.schemas
from tests.common_fixtures import aioresponses_mock

import framework.utils.auth.verifier


def test_verify_authorization(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    authorization_verification_input = (
        mlrun.common.schemas.AuthorizationVerificationInput(
            resource="/some-resource",
            action=mlrun.common.schemas.AuthorizationAction.create,
        )
    )

    async def _mock_successful_query_permissions(resource, action, *args):
        assert authorization_verification_input.resource == resource
        assert authorization_verification_input.action == action

    framework.utils.auth.verifier.AuthVerifier().query_permissions = (
        _mock_successful_query_permissions
    )
    response = client.post(
        "authorization/verifications", json=authorization_verification_input.dict()
    )
    assert response.status_code == http.HTTPStatus.OK.value


def test_authenticate_request_auth_info_basic(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
) -> None:
    mlrun.mlconf.httpdb.authentication.mode = "basic"
    mlrun.mlconf.httpdb.authentication.basic.username = "bugs"
    mlrun.mlconf.httpdb.authentication.basic.password = "bunny"
    authorization_verification_input = (
        mlrun.common.schemas.AuthorizationVerificationInput(
            resource="/some-resource",
            action=mlrun.common.schemas.AuthorizationAction.create,
        )
    )
    request_headers = {
        "authorization": "Basic YnVnczpidW5ueQ==",
        "cookie": "123",
    }

    async def _mock_successful_query_permissions(
        resource: str,
        action: mlrun.common.schemas.AuthorizationAction,
        auth_info: mlrun.common.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ):
        assert auth_info.username == "bugs"
        assert auth_info.password == "bunny"
        for key, value in request_headers.items():
            assert auth_info.request_headers[key] == value

    framework.utils.auth.verifier.AuthVerifier().query_permissions = (
        _mock_successful_query_permissions
    )
    response = client.post(
        "authorization/verifications",
        json=authorization_verification_input.dict(),
        headers=request_headers,
    )
    assert response.status_code == http.HTTPStatus.OK.value


def test_authenticate_request_auth_info_bearer(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
) -> None:
    mlrun.mlconf.httpdb.authentication.mode = "bearer"
    mlrun.mlconf.httpdb.authentication.bearer.token = "123"
    authorization_verification_input = (
        mlrun.common.schemas.AuthorizationVerificationInput(
            resource="/some-resource",
            action=mlrun.common.schemas.AuthorizationAction.create,
        )
    )
    request_headers = {
        "authorization": "Bearer 123",
    }

    async def _mock_successful_query_permissions(
        resource: str,
        action: mlrun.common.schemas.AuthorizationAction,
        auth_info: mlrun.common.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ):
        assert auth_info.token == "123"
        for key, value in request_headers.items():
            assert auth_info.request_headers[key] == value

    framework.utils.auth.verifier.AuthVerifier().query_permissions = (
        _mock_successful_query_permissions
    )
    response = client.post(
        "authorization/verifications",
        json=authorization_verification_input.dict(),
        headers=request_headers,
    )
    assert response.status_code == http.HTTPStatus.OK.value


def test_authenticate_request_auth_info_iguazio(
    api_url,
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    aioresponses_mock: aioresponses_mock,
) -> None:
    mlrun.mlconf.httpdb.authentication.mode = "iguazio"
    mock_request_headers = starlette.datastructures.Headers(
        {"cookie": "session=some-session-cookie"}
    )
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = mock_request_headers
    mock_response_headers = {
        "X-Remote-User": "username",
        "X-V3io-Session-Key": "session",
        "x-user-id": "123",
        "x-user-group-ids": "456",
        "x-v3io-session-planes": "control,data",
    }
    mock_request.state.request_id = "test-request-id"
    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"

    # Mock iguazio session verification endpoint
    def _verify_session_mock(*args, **kwargs):
        request_headers = kwargs["headers"]
        for header_key, header_value in mock_request_headers.items():
            assert request_headers[header_key] == header_value
        return aioresponses.CallbackResult(headers=mock_response_headers)

    aioresponses_mock.add(
        url,
        method="POST",
        callback=_verify_session_mock,
    )

    authorization_verification_input = (
        mlrun.common.schemas.AuthorizationVerificationInput(
            resource="/some-resource",
            action=mlrun.common.schemas.AuthorizationAction.create,
        )
    )

    async def _mock_successful_query_permissions(
        resource: str,
        action: mlrun.common.schemas.AuthorizationAction,
        auth_info: mlrun.common.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ):
        assert auth_info.username == mock_response_headers["X-Remote-User"]
        assert auth_info.session == mock_response_headers["X-V3io-Session-Key"]
        assert auth_info.user_id == mock_response_headers["x-user-id"]
        assert auth_info.user_group_ids == mock_response_headers[
            "x-user-group-ids"
        ].split(",")
        # we returned data in planes so a data session as well
        assert auth_info.data_session == mock_response_headers["X-V3io-Session-Key"]
        for key, value in mock_request_headers.items():
            assert auth_info.request_headers[key] == value

    framework.utils.auth.verifier.AuthVerifier().query_permissions = (
        _mock_successful_query_permissions
    )
    response = client.post(
        "authorization/verifications",
        json=authorization_verification_input.dict(),
        headers=mock_request_headers,
    )
    assert response.status_code == http.HTTPStatus.OK.value
