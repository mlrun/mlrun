# Copyright 2025 Iguazio
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

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt
import pytest

import mlrun.auth.utils
import mlrun.common.schemas
import mlrun.common.types
import mlrun.utils.logger
from mlrun.auth.providers import IGTokenProvider
from mlrun.config import config


@pytest.fixture
def encoded_jwt_token():
    iat = int(time.time())
    exp = iat + 1000
    payload = {
        "iat": iat,
        "exp": exp,
        "sub": "test-user",
        "typ": "Bearer",
        "preferred_username": "admin",
    }
    token = jwt.encode(payload, key=None, algorithm="none")
    return token, iat, exp


def test_ig_token_provider_successful_flow(encoded_jwt_token):
    encoded_jwt, iat, exp = encoded_jwt_token

    with patch.object(
        mlrun.auth.utils, "load_offline_token", return_value="offline-token"
    ):
        with patch("mlrun.utils.HTTPSessionWithRetry") as mock_session:
            mock_session_instance = mock_session.return_value
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"spec": {"accessToken": encoded_jwt}}
            mock_session_instance.request.return_value = mock_response

            provider = IGTokenProvider(token_endpoint="http://example.com")

            # Check token and lifetime
            assert provider.get_token()
            assert provider._token_total_lifetime == exp - iat
            assert provider._token_expiry_time == datetime.fromtimestamp(exp)


def test_with_empty_endpoint():
    with pytest.raises(mlrun.errors.MLRunValueError):
        IGTokenProvider(token_endpoint="")


@pytest.mark.parametrize(
    "threshold, total_lifetime, remaining_seconds, expected",
    [
        # Threshold None -> default 0.75
        (None, 100, 30, True),
        (None, 100, 20, False),
        # Threshold 0.5
        (0.5, 100, 60, True),
        (0.5, 100, 40, False),
        # Threshold 0.9
        (0.9, 100, 15, True),
        (0.9, 100, 5, False),
    ],
)
def test_is_access_token_valid(
    monkeypatch, threshold, total_lifetime, remaining_seconds, expected
):
    # create provider without __init__
    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._token = "token"
    provider._token_total_lifetime = total_lifetime
    provider._token_expiry_time = datetime.now() + timedelta(seconds=remaining_seconds)

    if threshold is not None:
        monkeypatch.setattr(
            config.auth_with_oauth_token, "refresh_threshold", threshold
        )
    # if threshold is None -> we don't patch, expect default 0.75

    assert provider._is_token_within_refresh_threshold() is expected


@pytest.mark.parametrize(
    "token, expected_lifetime, expected_expiration",
    [
        # Valid token with dynamic timestamps
        (
            jwt.encode(
                {
                    "iat": int((datetime.now() - timedelta(seconds=10)).timestamp()),
                    "exp": int((datetime.now() + timedelta(seconds=100)).timestamp()),
                },
                key="secret",
                algorithm="HS256",
            ),
            110,  # exp - iat
            datetime.fromtimestamp(
                int((datetime.now() + timedelta(seconds=100)).timestamp())
            ),
        ),
        # Missing iat
        (jwt.encode({"exp": 1100}, key="secret", algorithm="HS256"), 0, None),
        # Missing exp
        (jwt.encode({"iat": 1000}, key="secret", algorithm="HS256"), 0, None),
        # Empty token
        ("", 0, None),
        # Malformed token
        ("not-a-jwt", 0, None),
        # Incorrectly formatted JWT
        ("abc.def.ghi", 0, None),
    ],
)
def test_get_token_lifetime_and_expiry(token, expected_lifetime, expected_expiration):
    lifetime, expiry = IGTokenProvider._get_token_lifetime_and_expiry(token)
    assert lifetime == expected_lifetime
    if expected_lifetime > 0:
        # allow small delta for dynamic timestamp comparison
        assert abs((expiry - expected_expiration).total_seconds()) < 2
    else:
        assert expiry is None


def test_token_cleanup_when_expired(encoded_jwt_token, monkeypatch):
    token, iat, exp = encoded_jwt_token
    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._token = token
    provider._token_total_lifetime = 100
    provider._token_expiry_time = datetime.now() - timedelta(seconds=5)

    monkeypatch.setattr("mlrun.mlconf.auth_with_oauth_token.refresh_threshold", 0.5)

    assert not provider._is_token_within_refresh_threshold(cleanup_if_expired=True)
    assert provider._token is None
    assert provider._token_expiry_time is None


def test_post_fetch_hook_warns_near_expiry(monkeypatch, encoded_jwt_token):
    token, _, _ = encoded_jwt_token
    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._token = token
    provider._token_total_lifetime = 100
    provider._max_retries = 3
    provider._token_expiry_time = datetime.now() + timedelta(seconds=2)

    monkeypatch.setattr("mlrun.secrets.sync_secret_tokens", MagicMock())
    monkeypatch.setattr("mlrun.mlconf.auth_with_oauth_token.refresh_threshold", 0.9)
    # simulate the case where after fetch almost expired token still in the provider (due to the fetching failure)
    # post fetch hook in this case will make sure that token is not expired yet one more time
    provider._post_fetch_hook()
    assert provider._token == token


def test_post_fetch_hook_raises_if_no_token(monkeypatch):
    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._token = None
    provider._max_retries = 3
    monkeypatch.setattr("mlrun.secrets.sync_secret_tokens", MagicMock())
    # should detect empty token in post fetch hook and raise error
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        provider._post_fetch_hook()


def test_refresh_token_fails_and_is_not_valid(monkeypatch):
    provider = IGTokenProvider.__new__(IGTokenProvider)

    # expired token setup
    provider._token = "expired"
    provider._token_total_lifetime = 100
    provider._max_retries = 3

    provider._token_expiry_time = datetime.now() - timedelta(seconds=5)

    monkeypatch.setattr("mlrun.mlconf.auth_with_oauth_token.refresh_threshold", 0.5)
    monkeypatch.setattr("mlrun.mlconf.auth_with_oauth_token.token_file", "not-exists")

    # should fail during building request, because file with token doesn't exist
    # raises error because fetch failed, token expired, hence cleaned up
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        provider._refresh_token_if_needed()

    # ensure expired token was cleaned
    assert provider._token is None
    assert provider._token_expiry_time is None

    # set token lifetime to be almost expired
    provider._token = "almost_expired_token"
    provider._token_total_lifetime = 100
    provider._token_expiry_time = datetime.now() + timedelta(seconds=2)

    # refresh will fail, but should not raise an error in this case, because _token is not empty after post hook
    provider._refresh_token_if_needed()

    assert provider._token == "almost_expired_token"
    assert provider._token is not None
    assert provider._token_expiry_time is not None
