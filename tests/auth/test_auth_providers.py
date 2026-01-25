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
import mlrun.errors
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


def test_authenticated_user_id():
    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._token = jwt.encode(
        {"sub": "test-user"}, key="test-secret", algorithm="HS256"
    )
    assert provider.authenticated_user_id == "test-user"


@pytest.mark.parametrize(
    "runtime_kind,timeout,backoff,expect_timeout_retry",
    [
        # Not in runtime: use standard retry
        ("", 120, 10, False),
        # In runtime with timeout enabled: use timeout-based retry
        ("job", 120, 10, True),
        # In runtime but timeout disabled (0): use standard retry
        ("job", 0, 10, False),
    ],
)
def test_fetch_token_retry_strategy(
    monkeypatch, runtime_kind, timeout, backoff, expect_timeout_retry
):
    """Test that fetch_token uses the correct retry strategy based on runtime context."""
    monkeypatch.setenv("MLRUN_RUNTIME_KIND", runtime_kind)
    monkeypatch.setattr(
        "mlrun.mlconf.auth_with_oauth_token.runtime_token_refresh_timeout", timeout
    )
    monkeypatch.setattr(
        "mlrun.mlconf.auth_with_oauth_token.runtime_token_refresh_backoff", backoff
    )

    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._max_retries = 2

    with (
        patch("mlrun.utils.helpers.run_with_retry") as mock_run_with_retry,
        patch(
            "mlrun.utils.helpers.retry_until_successful"
        ) as mock_retry_until_successful,
    ):
        provider.fetch_token()

        if expect_timeout_retry:
            mock_retry_until_successful.assert_called_once()
            mock_run_with_retry.assert_not_called()
            # Verify timeout and backoff parameters from config
            call_kwargs = mock_retry_until_successful.call_args
            assert call_kwargs.kwargs["timeout"] == timeout
            assert call_kwargs.kwargs["backoff"] == backoff
        else:
            mock_run_with_retry.assert_called_once()
            mock_retry_until_successful.assert_not_called()


def test_runtime_retry_succeeds_after_initial_failures(encoded_jwt_token, monkeypatch):
    """
    Simulates the Kubelet propagation delay scenario:
    - First few attempts fail (old/invalid token in file)
    - Later attempts succeed (new token propagated)
    """
    encoded_jwt, _, _ = encoded_jwt_token

    monkeypatch.setenv("MLRUN_RUNTIME_KIND", "job")
    monkeypatch.setattr("mlrun.mlconf.httpdb.http.verify", True)
    # Use a timeout longer than backoff (10 seconds) to allow retries
    monkeypatch.setattr(
        "mlrun.mlconf.auth_with_oauth_token.runtime_token_refresh_timeout", 30
    )

    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._token = None
    provider._token_total_lifetime = 0
    provider._token_expiry_time = None
    provider._max_retries = 2
    provider._token_endpoint = "http://example.com"
    provider._timeout = 5

    # Track number of calls
    call_count = [0]

    def mock_load_offline_token(raise_on_error):
        call_count[0] += 1
        if call_count[0] <= 2:
            # First 2 calls return old/invalid token
            return "old-invalid-token"
        # Third call returns new valid token
        return "new-valid-token"

    monkeypatch.setattr("mlrun.auth.utils.load_offline_token", mock_load_offline_token)

    # Mock session to fail for old token, succeed for new token
    mock_session = MagicMock()

    def mock_request(**kwargs):
        request_body = kwargs.get("json", {})
        if request_body.get("refreshToken") == "new-valid-token":
            response = MagicMock()
            response.ok = True
            response.json.return_value = {"spec": {"accessToken": encoded_jwt}}
            return response
        else:
            # Raise an exception to trigger retry (simulating token endpoint rejection)
            raise mlrun.errors.MLRunRuntimeError("Invalid refresh token")

    mock_session.request.side_effect = mock_request
    provider._session = mock_session

    # This should succeed after retries
    provider.fetch_token()

    # Verify the token was set from the successful response
    assert provider._token == encoded_jwt
    # Verify multiple attempts were made (at least 3: 2 failures + 1 success)
    assert call_count[0] >= 3
