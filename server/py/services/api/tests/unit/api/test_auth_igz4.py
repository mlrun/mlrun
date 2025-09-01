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
import yaml

import mlrun.common.schemas
from mlrun.config import config
from mlrun.db.auth_utils import IGTokenProvider


def test_ig_token_provider_successful_flow():
    # Generate dynamic iat and exp
    iat = int(time.time())
    exp = iat + 1000
    token_payload = {
        "iat": iat,
        "exp": exp,
        "sub": "21c5db16-f0ca-4951-9702-3208a022bad6",
        "typ": "Bearer",
        "preferred_username": "admin",
    }

    encoded_jwt = jwt.encode(token_payload, key=None, algorithm="none")

    with patch.object(
        IGTokenProvider, "_load_offline_token", return_value="offline-token"
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


def test_get_offline_token_from_env(monkeypatch):
    monkeypatch.setenv("MLRUN_AUTH_OFFLINE_TOKEN", "env-token")
    token = IGTokenProvider._get_offline_token_from_env()
    assert token == "env-token"
    monkeypatch.delenv("MLRUN_AUTH_OFFLINE_TOKEN", raising=False)
    assert IGTokenProvider._get_offline_token_from_env() is None


@pytest.mark.parametrize(
    "data, token_name, expected_token",
    [
        # 1. Valid default token
        (
            {"secretTokens": [{"name": "default", "token": "file-token"}]},
            None,
            "file-token",
        ),
        # 2. Valid token with custom name
        (
            {"secretTokens": [{"name": "custom", "token": "custom-token"}]},
            "custom",
            "custom-token",
        ),
        # 3. secretTokens missing
        (
            {},
            None,
            None,
        ),
        # 4. secretTokens not a list
        (
            {"secretTokens": "not-a-list"},
            None,
            None,
        ),
        # 5. secretTokens empty list
        (
            {"secretTokens": []},
            None,
            None,
        ),
        # 6. Multiple matching tokens
        (
            {
                "secretTokens": [
                    {"name": "default", "token": "t1"},
                    {"name": "default", "token": "t2"},
                ]
            },
            None,
            None,
        ),
        # 7. Token entry missing 'token' field
        (
            {"secretTokens": [{"name": "default"}]},
            None,
            None,
        ),
        # 8. Empty default token name, no default, use 1st token
        (
            {
                "secretTokens": [
                    {"name": "token1", "token": "file-token1"},
                    {"name": "token2", "token": "file-token2"},
                ]
            },
            None,
            "file-token1",
        ),
    ],
)
def test_parse_offline_token_data_cases(data, token_name, expected_token, monkeypatch):
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.auth_token_name", token_name
    )

    token_file = "/fake/path.yaml"

    # Suppress raising errors, we just check return value
    token = IGTokenProvider._parse_offline_token_data(
        data, token_file, raise_on_error=False
    )
    assert token == expected_token


@pytest.mark.parametrize(
    "data, token_name",
    [
        # secretTokens missing
        ({}, None),
        # secretTokens not a list
        ({"secretTokens": "not-a-list"}, None),
        # secretTokens empty
        ({"secretTokens": []}, None),
        # Multiple matching tokens
        (
            {
                "secretTokens": [
                    {"name": "default", "token": "t1"},
                    {"name": "default", "token": "t2"},
                ]
            },
            None,
        ),
        # Token entry missing 'token'
        ({"secretTokens": [{"name": "default"}]}, None),
    ],
)
def test_parse_offline_token_data_raise_exception(data, token_name, monkeypatch):
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.auth_token_name", token_name
    )

    token_file = "/fake/path.yaml"

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        IGTokenProvider._parse_offline_token_data(data, token_file, raise_on_error=True)


def test_with_empty_endpoint():
    with pytest.raises(mlrun.errors.MLRunValueError):
        IGTokenProvider(token_endpoint="")


@pytest.mark.parametrize(
    "env_token, file_token, expected",
    [
        # env token exists
        ("env-token", None, "env-token"),
        # only file token exists
        (None, "file-token", "file-token"),
        # token missing
        (None, None, None),
    ],
)
def test_load_offline_token_parametrized(env_token, file_token, expected):
    # create provider without __init__
    provider = IGTokenProvider.__new__(IGTokenProvider)

    with (
        patch.object(provider, "_get_offline_token_from_env", return_value=env_token),
        patch.object(provider, "_get_offline_token_from_file", return_value=file_token),
    ):
        token = provider._load_offline_token()
        assert token == expected


def test_token_file_not_exists(monkeypatch):
    fake_file = "no_such_file.yaml"
    monkeypatch.setattr(config.auth_with_oauth_token, "auth_token_file", str(fake_file))
    # create provider without __init__
    provider = IGTokenProvider.__new__(IGTokenProvider)

    result = provider._get_offline_token_from_file(raise_on_error=False)
    assert result is None

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        provider._get_offline_token_from_file(raise_on_error=True)


@pytest.mark.parametrize(
    "threshold, total_lifetime, remaining_seconds, expected",
    [
        # Threshold None -> default 0.75
        (None, 100, 80, True),
        (None, 100, 60, False),
        # Threshold 0.5
        (0.5, 100, 60, True),
        (0.5, 100, 40, False),
        # Threshold 0.9
        (0.9, 100, 95, True),
        (0.9, 100, 85, False),
    ],
)
def test_is_access_token_valid(
    monkeypatch, threshold, total_lifetime, remaining_seconds, expected
):
    # create provider without __init__
    provider = IGTokenProvider.__new__(IGTokenProvider)
    provider._access_token = "token"
    provider._token_total_lifetime = total_lifetime
    provider._token_expiry_time = datetime.now() + timedelta(seconds=remaining_seconds)

    if threshold is not None:
        monkeypatch.setattr(
            config.auth_with_oauth_token, "refresh_threshold", threshold
        )
    # if threshold is None -> we don't patch, expect default 0.75

    assert provider._is_access_token_valid() is expected


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
    lifetime, expiry = IGTokenProvider.get_token_lifetime_and_expiry(token)
    assert lifetime == expected_lifetime
    if expected_lifetime > 0:
        # allow small delta for dynamic timestamp comparison
        assert abs((expiry - expected_expiration).total_seconds()) < 2
    else:
        assert expiry is None


@pytest.mark.parametrize(
    "file_content, expected_token, raise_on_error",
    [
        # Valid token file with default name
        ({"secretTokens": [{"name": "default", "token": "abc123"}]}, "abc123", True),
        # Valid token file with custom name
        ({"secretTokens": [{"name": "custom", "token": "xyz789"}]}, "xyz789", True),
        # Missing token field
        ({"secretTokens": [{"name": "default"}]}, None, False),
        # Empty secretTokens list
        ({"secretTokens": []}, None, False),
        # Invalid secretTokens type
        ({"secretTokens": "not-a-list"}, None, False),
        # Malformed YAML case (special marker)
        ("__MALFORMED__", None, True),
    ],
)
def test_get_offline_token_from_file(
    tmp_path, monkeypatch, file_content, expected_token, raise_on_error
):
    token_file = tmp_path / "token.yaml"

    # Write content to file
    if file_content == "__MALFORMED__":
        # Write invalid YAML
        token_file.write_text("invalid: [unbalanced brackets")
    else:
        with open(token_file, "w") as f:
            yaml.safe_dump(file_content, f)

    # Monkeypatch config to point to temp file
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.auth_token_file", str(token_file)
    )

    # Create IGTokenProvider instance without calling __init__
    provider = IGTokenProvider.__new__(IGTokenProvider)

    if expected_token is None and raise_on_error:
        # Expect MLRunRuntimeError
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            provider._get_offline_token_from_file(raise_on_error=True)
    else:
        token = provider._get_offline_token_from_file(raise_on_error=raise_on_error)
        assert token == expected_token
