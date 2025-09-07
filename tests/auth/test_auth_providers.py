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
from mlrun.auth.providers import IGTokenProvider
from mlrun.config import config


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
    provider._token = "token"
    provider._token_total_lifetime = total_lifetime
    provider._token_expiry_time = datetime.now() + timedelta(seconds=remaining_seconds)

    if threshold is not None:
        monkeypatch.setattr(
            config.auth_with_oauth_token, "refresh_threshold", threshold
        )
    # if threshold is None -> we don't patch, expect default 0.75

    assert provider._is_token_valid() is expected


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
