# Copyright 2024 Iguazio
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

import base64
from unittest.mock import MagicMock, patch

import pytest

import mlrun
import mlrun.auth.nuclio
import mlrun.common.schemas
import mlrun.common.schemas.auth
import mlrun.common.types
import mlrun.runtimes.nuclio


@pytest.mark.parametrize(
    "host, path, expected_url",
    [
        ("example.com", "/api", "example.com/api"),
        ("example.com/", "/api", "example.com/api"),
        ("example.com", "api", "example.com/api"),
        ("example.com/", "api", "example.com/api"),
        ("example.com/", "/api/", "example.com/api"),
        ("example.com/", "/api/long/path/", "example.com/api/long/path"),
        ("example.com", "/", "example.com"),
        ("example.com/", "/", "example.com"),
        ("example.com", None, "example.com"),
        ("example.com/", None, "example.com"),
    ],
)
def test_get_invoke_url(host, path, expected_url):
    # testing server side api gateway
    api_gateway = mlrun.common.schemas.APIGateway(
        metadata=mlrun.common.schemas.APIGatewayMetadata(name="test"),
        spec=mlrun.common.schemas.APIGatewaySpec(
            name="test", host=host, path=path, upstreams=[]
        ),
    )
    assert api_gateway.get_invoke_url() == expected_url

    # testing client side api gateway
    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(name="test"),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            project="test", host=host, path=path, functions=["test"]
        ),
    )
    assert api_gateway.invoke_url == "https://" + expected_url


def test_with_annotations():
    annotations = {"key1": "value1", "key2": "value2"}

    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(name="test"),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            project="test", host="host", path="path", functions=["test"]
        ),
    )

    api_gateway.with_annotations(annotations)
    assert api_gateway.metadata.annotations == annotations


@pytest.mark.parametrize(
    "headers, expected_token, expected_username, expected_password",
    [
        # Bearer token auth
        (
            {"Authorization": "Bearer my-bearer-token"},
            "my-bearer-token",
            None,
            None,
        ),
        # Basic auth
        (
            {"Authorization": f"Basic {base64.b64encode(b'user:pass').decode()}"},
            None,
            "user",
            "pass",
        ),
        # Basic auth with colon in password
        (
            {
                "Authorization": f"Basic {base64.b64encode(b'user:pass:with:colons').decode()}"
            },
            None,
            "user",
            "pass:with:colons",
        ),
        # Empty headers
        ({}, None, None, None),
        # None headers
        (None, None, None, None),
        # Headers without Authorization
        ({"Content-Type": "application/json"}, None, None, None),
    ],
)
def test_from_request_headers(
    headers, expected_token, expected_username, expected_password
):
    """Test NuclioAuthInfo.from_request_headers with various header formats"""
    auth_info = mlrun.auth.nuclio.NuclioAuthInfo.from_request_headers(headers)

    if expected_token:
        assert auth_info._token == expected_token
    else:
        assert auth_info._token is None

    if expected_username:
        assert auth_info._username == expected_username
        assert auth_info._password == expected_password


def test_from_envvar_iguazio_v4_mode(monkeypatch):
    """Test NuclioAuthInfo.from_envvar in iguazio v4 mode"""
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication,
        "mode",
        mlrun.common.types.AuthenticationMode.IGUAZIO_V4,
    )
    monkeypatch.setattr(
        mlrun.mlconf,
        "auth_token_endpoint",
        "https://iguazio.example/api/v1/authentication/refresh-access-token",
    )

    with patch(
        "mlrun.auth.providers.IGTokenProvider",
    ) as mock_token_provider_cls:
        mock_token_provider = MagicMock()
        mock_token_provider.get_token.return_value = "access-v4-token"
        mock_token_provider_cls.return_value = mock_token_provider

        auth_info = mlrun.auth.nuclio.NuclioAuthInfo.from_envvar()

        mock_token_provider_cls.assert_called_once_with(
            token_endpoint="https://iguazio.example/api/v1/authentication/refresh-access-token"
        )
        mock_token_provider.get_token.assert_called_once()
        assert auth_info._token == "access-v4-token"


def test_from_envvar_non_iguazio_v4_mode(monkeypatch):
    """Test NuclioAuthInfo.from_envvar in non-iguazio v4 mode"""
    monkeypatch.setenv("V3IO_ACCESS_KEY", "test-access-key")
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication,
        "mode",
        "none",  # Any non-IGUAZIO_V4 value
    )
    auth_info = mlrun.auth.nuclio.NuclioAuthInfo.from_envvar()
    # Should call parent's from_envvar which uses V3IO_ACCESS_KEY
    assert auth_info._token is None
    # Parent class stores access_key as password
    assert auth_info._password == "test-access-key"


def test_to_requests_auth_with_token():
    """Test NuclioAuthInfo.to_requests_auth returns bearer auth when token is set"""
    auth_info = mlrun.auth.nuclio.NuclioAuthInfo(token="my-bearer-token")
    auth = auth_info.to_requests_auth()

    # Verify it's a valid auth object
    assert auth is not None

    # Verify it adds Bearer token to request headers
    mock_request = MagicMock()
    mock_request.headers = {}
    auth.__call__(mock_request)

    assert "Authorization" in mock_request.headers
    assert mock_request.headers["Authorization"] == "Bearer my-bearer-token"


def test_to_requests_auth_without_token():
    """Test NuclioAuthInfo.to_requests_auth falls back to parent when no token"""
    auth_info = mlrun.auth.nuclio.NuclioAuthInfo(username="user", password="pass")
    auth = auth_info.to_requests_auth()

    # Parent class should return basic auth
    assert auth is not None


def test_to_nuclio_auth_info_iguazio_v4_mode_with_bearer(monkeypatch):
    """Test AuthInfo.to_nuclio_auth_info in iguazio v4 mode with bearer token"""
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication,
        "mode",
        mlrun.common.types.AuthenticationMode.IGUAZIO_V4,
    )
    auth_info = mlrun.common.schemas.auth.AuthInfo(
        request_headers={"Authorization": "Bearer my-v4-token"}
    )
    nuclio_auth = mlrun.auth.nuclio.NuclioAuthInfo.from_auth_info(auth_info)

    assert nuclio_auth is not None
    assert nuclio_auth._token == "my-v4-token"


def test_to_nuclio_auth_info_iguazio_v4_mode_with_basic(monkeypatch):
    """Test AuthInfo.to_nuclio_auth_info in iguazio v4 mode with basic auth"""
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication,
        "mode",
        mlrun.common.types.AuthenticationMode.IGUAZIO_V4,
    )
    encoded_creds = base64.b64encode(b"admin:password123").decode()
    auth_info = mlrun.common.schemas.auth.AuthInfo(
        request_headers={"Authorization": f"Basic {encoded_creds}"}
    )
    nuclio_auth = mlrun.auth.nuclio.NuclioAuthInfo.from_auth_info(auth_info)

    assert nuclio_auth is not None
    assert nuclio_auth._username == "admin"
    assert nuclio_auth._password == "password123"


def test_to_nuclio_auth_info_non_v4_mode_with_session(monkeypatch):
    """Test AuthInfo.to_nuclio_auth_info in non-v4 mode with session"""
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication,
        "mode",
        "none",  # Any non-IGUAZIO_V4 value
    )
    auth_info = mlrun.common.schemas.auth.AuthInfo(session="my-session-token")
    nuclio_auth = mlrun.auth.nuclio.NuclioAuthInfo.from_auth_info(auth_info)

    assert nuclio_auth is not None
    assert nuclio_auth._password == "my-session-token"


def test_to_nuclio_auth_info_non_v4_mode_no_session(monkeypatch):
    """Test AuthInfo.to_nuclio_auth_info returns None when no auth in non-v4 mode"""
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication,
        "mode",
        "none",  # Any non-IGUAZIO_V4 value
    )
    auth_info = mlrun.common.schemas.auth.AuthInfo(session="")
    nuclio_auth = mlrun.auth.nuclio.NuclioAuthInfo.from_auth_info(auth_info)

    assert nuclio_auth is None


def test_invoke_basic_auth_creates_nuclio_auth_info(monkeypatch):
    """Test that invoke with basic auth uses NuclioAuthInfo correctly"""
    api_gateway = _create_api_gateway("basic")

    # Mock the requests.request
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.request", return_value=mock_response) as mock_request:
        api_gateway.invoke(
            method="GET", credentials=("test-user", "test-pass"), path="/test"
        )

        # Verify request was made with auth
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["auth"] is not None


def test_invoke_access_key_auth_with_credentials(monkeypatch):
    """Test invoke with access_key auth and explicit credentials"""
    api_gateway = _create_api_gateway("access_key")

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.request", return_value=mock_response) as mock_request:
        api_gateway.invoke(
            method="GET", credentials=("user", "access-key"), path="/test"
        )

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["auth"] is not None


def test_invoke_access_key_auth_from_envvar(monkeypatch):
    """Test invoke with access_key auth using environment variable"""
    api_gateway = _create_api_gateway("access_key")

    mock_response = MagicMock()
    mock_response.status_code = 200

    # Mock from_envvar to return auth with token
    with patch.object(
        mlrun.auth.nuclio.NuclioAuthInfo,
        "from_envvar",
        return_value=mlrun.auth.nuclio.NuclioAuthInfo(token="env-access-token"),
    ):
        with patch("requests.request", return_value=mock_response) as mock_request:
            api_gateway.invoke(method="GET", path="/test")

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["auth"] is not None


def test_invoke_access_key_auth_no_credentials_raises(monkeypatch):
    """Test invoke with access_key auth raises when no credentials available"""
    api_gateway = _create_api_gateway("access_key")

    # Mock from_envvar to return None (no auth available)
    with patch.object(
        mlrun.auth.nuclio.NuclioAuthInfo,
        "from_envvar",
        return_value=mlrun.auth.nuclio.NuclioAuthInfo(),
    ):
        with patch.object(
            mlrun.auth.nuclio.NuclioAuthInfo,
            "to_requests_auth",
            return_value=None,
        ):
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
                api_gateway.invoke(method="GET", path="/test")

            assert "V3IO_ACCESS_KEY" in str(exc_info.value)


def test_invoke_iguazio_auth_uses_from_envvar(monkeypatch):
    """Test invoke with iguazio auth uses from_envvar directly"""
    api_gateway = _create_api_gateway("iguazio")

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_auth_result = MagicMock()
    with patch.object(
        mlrun.auth.nuclio.NuclioAuthInfo, "from_envvar"
    ) as mock_from_envvar:
        mock_nuclio_auth = MagicMock()
        mock_nuclio_auth.to_requests_auth.return_value = mock_auth_result
        mock_from_envvar.return_value = mock_nuclio_auth

        with patch("requests.request", return_value=mock_response) as mock_request:
            api_gateway.invoke(method="GET", path="/test")

            # Verify from_envvar was called as a classmethod (no instance)
            mock_from_envvar.assert_called_once()
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["auth"] == mock_auth_result


def test_invoke_basic_auth_requires_credentials():
    """Test that basic auth invoke requires credentials"""
    api_gateway = _create_api_gateway("basic")

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
        api_gateway.invoke(method="GET", path="/test")

    assert "authentication" in str(exc_info.value).lower()
    assert "credentials" in str(exc_info.value).lower()


def _create_api_gateway(authentication_mode="none"):
    """Helper to create an API gateway with specified auth mode"""
    auth_map = {
        "none": mlrun.runtimes.nuclio.api_gateway.NoneAuth(),
        "basic": mlrun.runtimes.nuclio.api_gateway.BasicAuth(
            username="user", password="pass"
        ),
        "access_key": mlrun.runtimes.nuclio.api_gateway.AccessKeyAuth(),
        "iguazio": mlrun.runtimes.nuclio.api_gateway.IguazioAuth(),
    }

    return mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(name="test"),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            project="test",
            host="https://example.com",
            path="/api",
            functions=["test-func"],
            authentication=auth_map[authentication_mode],
        ),
        status=mlrun.runtimes.nuclio.api_gateway.APIGatewayStatus(
            state=mlrun.common.schemas.APIGatewayState.ready
        ),
    )
