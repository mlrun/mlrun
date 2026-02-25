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


import re
import textwrap
import time
from unittest.mock import patch

import jwt
import pytest
import yaml

import mlrun.auth.utils
import mlrun.common.schemas
import mlrun.errors
from mlrun.config import config


def _create_jwt_token(payload: dict, add_defaults: bool = True) -> str:
    """Helper to create a JWT token with a given payload.

    :param payload: The payload to encode in the JWT.
    :param add_defaults: If True, adds default 'exp' and 'sub' if not present.
                        Set to False when testing missing claims.
    """
    if add_defaults:
        if "exp" not in payload:
            payload["exp"] = time.time() + 3600
        if "sub" not in payload:
            payload["sub"] = "test-user"

    return jwt.encode(payload, key="test-secret", algorithm="HS256")


def test_get_offline_token_from_env(monkeypatch):
    monkeypatch.setenv("MLRUN_AUTH_OFFLINE_TOKEN", "env-token")
    token = mlrun.auth.utils.get_offline_token_from_env()
    assert token == "env-token"
    monkeypatch.delenv("MLRUN_AUTH_OFFLINE_TOKEN", raising=False)
    assert mlrun.auth.utils.get_offline_token_from_env() is None


@pytest.mark.parametrize(
    "data, token_name, expected_token, expected_name",
    [
        # 1. Valid default token
        (
            [{"name": "default", "token": "file-token"}],
            None,
            "file-token",
            "default",
        ),
        # 2. Valid token with custom name
        (
            [{"name": "custom", "token": "custom-token"}],
            "custom",
            "custom-token",
            "custom",
        ),
        # # 3. secretTokens not a list
        ("not-a-list", None, None, None),
        # # 4. secretTokens empty list
        ([], None, None, None),
        # 5. Multiple matching tokens
        (
            [
                {"name": "default", "token": "t1"},
                {"name": "default", "token": "t2"},
            ],
            None,
            None,
            None,
        ),
        # 6. Token entry missing 'token' field
        ([{"name": "default"}], None, None, None),
        # 7. Empty default token name, no default, use 1st token
        (
            [
                {"name": "token1", "token": "file-token1"},
                {"name": "token2", "token": "file-token2"},
            ],
            None,
            "file-token1",
            "token1",
        ),
    ],
)
def test_parse_offline_token_data_cases(
    data, token_name, expected_token, expected_name, monkeypatch
):
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.token_name", token_name
    )
    # Suppress raising errors, we just check return value
    token, name = mlrun.auth.utils.parse_offline_token_data(data, raise_on_error=False)
    assert name == expected_name
    assert token == expected_token


@pytest.mark.parametrize(
    "data, token_name",
    [
        # secretTokens not a list
        ("not-a-list", None),
        # secretTokens empty
        ([], None),
        # Multiple matching tokens
        (
            [
                {"name": "default", "token": "t1"},
                {"name": "default", "token": "t2"},
            ],
            None,
        ),
        # Token entry missing 'token'
        ([{"name": "default"}], None),
    ],
)
def test_parse_offline_token_data_raise_exception(data, token_name, monkeypatch):
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.token_name", token_name
    )

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        mlrun.auth.utils.parse_offline_token_data(data, raise_on_error=True)


@pytest.mark.parametrize(
    "env_token, file_token, expected",
    [
        # env token exists
        ("env-token", None, ("env-token", "default")),
        # only file token exists
        (None, ("file-token", "default"), ("file-token", "default")),
        # token missing
        (None, (None, None), (None, None)),
    ],
)
def test_load_offline_token_parametrized(env_token, file_token, expected, monkeypatch):
    monkeypatch.setattr(config.auth_with_oauth_token, "token_name", None)
    with (
        patch.object(
            mlrun.auth.utils, "get_offline_token_from_env", return_value=env_token
        ),
        patch.object(
            mlrun.auth.utils,
            "get_offline_token_from_file",
            return_value=file_token,
        ),
    ):
        token, _ = mlrun.auth.utils.load_offline_token(raise_on_error=False)
        assert token == expected[0]


def test_token_file_not_exists(monkeypatch):
    fake_file = "no_such_file.yaml"
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", str(fake_file))

    result, _ = mlrun.auth.utils.get_offline_token_from_file(raise_on_error=False)
    assert result is None

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        mlrun.auth.utils.get_offline_token_from_file(raise_on_error=True)


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
        if isinstance(file_content, dict):
            yaml.safe_dump(file_content, token_file.open("w"))
        else:
            token_file.write_text(file_content)

    # Monkeypatch config to point to temp file
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.token_file", str(token_file)
    )

    if expected_token is None and raise_on_error:
        # Expect MLRunRuntimeError
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            mlrun.auth.utils.get_offline_token_from_file(raise_on_error=True)
    else:
        token, _ = mlrun.auth.utils.get_offline_token_from_file(
            raise_on_error=raise_on_error
        )
        assert token == expected_token


@pytest.mark.parametrize(
    "token_user_ids, auth_user_id, expected_count",
    [
        # Case 1: one token, returns 1 token (same user)
        (["test-user-123"], "test-user-123", 1),
        # Case 2: two tokens, returns 2 tokens (same user)
        (["test-user-123", "test-user-123"], "test-user-123", 2),
        # Case 3: two tokens, return 1 token (one different user)
        (["test-user-123", "other-user"], "test-user-123", 1),
        # Case 4: two tokens, return 0 tokens (both different users)
        (["other-user-1", "other-user-2"], "test-user-123", 0),
    ],
)
def test_load_and_prepare_secret_tokens_valid(
    tmp_path, token_user_ids, auth_user_id, expected_count, monkeypatch
):
    # Generate valid JWT tokens with the specified user IDs
    tokens = []
    for idx, user_id in enumerate(token_user_ids):
        jwt_token = _create_jwt_token({"sub": user_id, "exp": 9999999999})
        tokens.append({"name": f"token{idx + 1}", "token": jwt_token})

    content = {"secretTokens": tokens}
    path = _write_file(tmp_path, "tokens.yml", content)
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", path)

    secret_tokens = mlrun.auth.utils.load_and_prepare_secret_tokens(
        auth_user_id=auth_user_id
    )
    assert isinstance(secret_tokens, list)
    assert len(secret_tokens) == expected_count
    assert all(isinstance(t, mlrun.common.schemas.SecretToken) for t in secret_tokens)


@pytest.mark.parametrize(
    "content",
    [
        textwrap.dedent("""\
            notSecretTokens:
              - name: token1
                token: abc123
        """),
        textwrap.dedent("""\
            secretTokens: []
        """),
        textwrap.dedent("""\
            secretTokens:
              token1: abc123
        """),
    ],
)
def test_load_secret_tokens_from_file_invalid(tmp_path, content, monkeypatch):
    path = _write_file(tmp_path, "tokens.yml", content)
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", path)
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        mlrun.auth.utils.load_secret_tokens_from_file()


@pytest.mark.parametrize(
    "secret_tokens_factory, expected_error",
    [
        # Missing/empty token name
        (
            lambda: [
                mlrun.common.schemas.SecretToken(
                    name="",
                    token=_create_jwt_token({"sub": "user-123", "exp": 9999999999}),
                ),
            ],
            mlrun.errors.MLRunInvalidArgumentError,
        ),
        # Duplicate token names
        (
            lambda: [
                mlrun.common.schemas.SecretToken(
                    name="dup",
                    token=_create_jwt_token({"sub": "user-123", "exp": 9999999999}),
                ),
                mlrun.common.schemas.SecretToken(
                    name="dup",
                    token=_create_jwt_token({"sub": "user-123", "exp": 9999999999}),
                ),
            ],
            mlrun.errors.MLRunInvalidArgumentError,
        ),
        # Invalid JWT token (not a valid JWT)
        (
            lambda: [
                mlrun.common.schemas.SecretToken(
                    name="invalid_jwt",
                    token="not-a-valid-jwt-token",
                ),
            ],
            mlrun.errors.MLRunInvalidArgumentError,
        ),
    ],
)
def test_validate_secret_tokens_invalid_entries(secret_tokens_factory, expected_error):
    secret_tokens = secret_tokens_factory()
    with pytest.raises(expected_error):
        mlrun.auth.utils.extract_and_validate_tokens_info(
            secret_tokens, authenticated_id="user-123", filter_by_authenticated_id=False
        )


def test_read_secret_tokens_file_non_existent(tmp_path, monkeypatch):
    file_path = tmp_path / "does_not_exist.yml"
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", str(file_path))

    result = mlrun.auth.utils.read_secret_tokens_file(raise_on_error=False)
    assert result is None

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        mlrun.auth.utils.read_secret_tokens_file(raise_on_error=True)


@pytest.mark.parametrize(
    "file_name, file_content, raise_on_error, expect_error, expected_result",
    [
        # 1. Empty file
        ("tokens.yaml", "", True, True, None),
        ("tokens.yaml", "", False, False, None),
        # 2. Non-dict YAML (list at root)
        ("tokens.yaml", "- just-a-list-item", True, True, None),
        ("tokens.yaml", "- just-a-list-item", False, False, None),
        # 3. Malformed YAML → yaml.safe_load will throw error
        ("tokens.yaml", "::: bad yaml :::", True, True, None),
        ("tokens.yaml", "::: bad yaml :::", False, False, None),
        # 4. Valid YAML regardless of extension
        (
            "tokens.txt",
            {"secretTokens": [{"name": "n1", "token": "t1"}]},
            True,
            False,
            {"secretTokens": [{"name": "n1", "token": "t1"}]},
        ),
    ],
)
def test_read_secret_tokens_file(
    tmp_path,
    monkeypatch,
    file_name,
    file_content,
    raise_on_error,
    expect_error,
    expected_result,
):
    # Use shared helper to write file
    file_path = _write_file(tmp_path, file_name, file_content)

    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", str(file_path))

    if expect_error and raise_on_error:
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            mlrun.auth.utils.read_secret_tokens_file(raise_on_error=raise_on_error)
    else:
        result = mlrun.auth.utils.read_secret_tokens_file(raise_on_error=raise_on_error)
        assert result == expected_result


def _write_file(tmp_path, name: str, content) -> str:
    file_path = tmp_path / name
    if isinstance(content, dict):
        yaml.safe_dump(content, file_path.open("w"))
    else:
        file_path.write_text(content)
    return str(file_path)


@pytest.mark.parametrize(
    "token_1, token_2, should_raise, expected_err_msg, expected_token_1, expected_token_2, authenticated_id",
    [
        # Valid tokens with different names
        (
            {
                "token_name": "token1",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            {
                "token_name": "token2",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            False,
            None,
            {"sub": "user-123", "exp": 9999999999},
            {"sub": "user-123", "exp": 9999999999},
            "user-123",
        ),
        # Missing expiration claim
        (
            {
                "token_name": "token1",
                "token_payload": {"sub": "user-123"},
                "add_defaults": False,
            },
            {
                "token_name": "token2",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            True,
            "Offline token 'token1' is missing the 'exp' (expiration) claim",
            None,
            None,
            "user-123",
        ),
        # Missing subject claim
        (
            {
                "token_name": "token1",
                "token_payload": {"exp": 9999999999},
                "add_defaults": False,
            },
            {
                "token_name": "token2",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            True,
            "Offline token 'token1' is missing the 'sub' (subject) claim",
            None,
            None,
            "user-123",
        ),
        # Token from wrong user (not matching authenticated ID)
        (
            {
                "token_name": "token1",
                "token_payload": {"sub": "different-user", "exp": 9999999999},
                "add_defaults": True,
            },
            {
                "token_name": "token2",
                "token_payload": {"sub": "different-user", "exp": 9999999999},
                "add_defaults": True,
            },
            True,
            "Offline token 'token1' does not match the authenticated user ID. Stored tokens can only belong to the"
            " authenticated user.",
            None,
            None,
            "user-123",
        ),
        # Duplicate token names
        (
            {
                "token_name": "token1",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            {
                "token_name": "token1",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            True,
            "Invalid or duplicate token name 'token1' found in request payload",
            None,
            None,
            "user-123",
        ),
        # Missing token name
        (
            {
                "token_name": "",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            {
                "token_name": "token2",
                "token_payload": {"sub": "user-123", "exp": 9999999999},
                "add_defaults": True,
            },
            True,
            "Invalid or duplicate token name '' found in request payload",
            None,
            None,
            "user-123",
        ),
    ],
)
def test_extract_and_validate_tokens_info(
    token_1,
    token_2,
    should_raise,
    expected_err_msg,
    expected_token_1,
    expected_token_2,
    authenticated_id,
):
    secret_tokens = [
        mlrun.common.schemas.SecretToken(
            name=token_1["token_name"],
            token=_create_jwt_token(
                token_1["token_payload"], add_defaults=token_1.get("add_defaults", True)
            ),
        ),
        mlrun.common.schemas.SecretToken(
            name=token_2["token_name"],
            token=_create_jwt_token(
                token_2["token_payload"], add_defaults=token_2.get("add_defaults", True)
            ),
        ),
    ]

    if should_raise:
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match=re.escape(expected_err_msg)
        ):
            mlrun.auth.utils.extract_and_validate_tokens_info(
                secret_tokens, authenticated_id
            )
    else:
        tokens_info = mlrun.auth.utils.extract_and_validate_tokens_info(
            secret_tokens, authenticated_id
        )
        assert tokens_info["token1"]["token_exp"] == expected_token_1["exp"]
        assert tokens_info["token2"]["token_exp"] == expected_token_2["exp"]


@pytest.mark.parametrize(
    "tokens, auth_user_id, expected_names",
    [
        # Case 1: 2 tokens, returns 1 token for matching user
        (
            lambda: [
                {"name": "admin", "token": _create_jwt_token({"sub": "admin-user"})},
                {
                    "name": "normal-user",
                    "token": _create_jwt_token({"sub": "normal-user"}),
                },
            ],
            "normal-user",
            ["normal-user"],
        ),
        # Case 2: 2 tokens, returns 2 tokens - no auth_user_id given (None)
        (
            lambda: [
                {"name": "admin", "token": _create_jwt_token({"sub": "admin-user"})},
                {
                    "name": "normal-user",
                    "token": _create_jwt_token({"sub": "normal-user"}),
                },
            ],
            None,
            [],
        ),
        # Case 3: 1 token, returns 0 tokens for non-matching user
        (
            lambda: [
                {"name": "admin", "token": _create_jwt_token({"sub": "admin-user"})},
            ],
            "different-user",
            [],
        ),
    ],
)
def test_validate_secret_tokens_filters_by_auth_user(
    tokens, auth_user_id, expected_names, monkeypatch
):
    """Test that validate_secret_tokens filters tokens by auth_user_id (JWT 'sub' claim)."""
    # Set a dummy token file path for the function
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", "/tmp/dummy.yml")

    tokens_list = [
        mlrun.common.schemas.SecretToken(
            name=token["name"],
            token=token["token"],
        )
        for token in tokens()
    ]
    result = mlrun.auth.utils.extract_and_validate_tokens_info(
        tokens_list, authenticated_id=auth_user_id, filter_by_authenticated_id=True
    )

    assert list(result.keys()) == expected_names


@pytest.mark.parametrize(
    "token, add_defaults, expected_sub",
    [
        ({}, False, None),  # Empty payload without defaults -> no 'sub' claim
        ({"sub": "user-123"}, False, "user-123"),  # Explicit 'sub' claim
    ],
)
def test_resolve_jwt_subject(token, add_defaults, expected_sub):
    """Test extracting 'sub' claim from JWT token."""
    jwt_token = _create_jwt_token(token, add_defaults=add_defaults)
    result = mlrun.auth.utils.resolve_jwt_subject(jwt_token, raise_on_error=True)
    assert result == expected_sub


@pytest.mark.parametrize(
    "exp_offset, buffer_seconds, should_expire",
    [
        # Token expired 10 seconds ago, no buffer
        (-10, 0, True),
        # Token expires in 10 seconds, no buffer
        (10, 0, False),
        # Token expires in 10 seconds, buffer 15 seconds (should be expired)
        (10, 15, True),
        # Token expires in 10 seconds, buffer 5 seconds (should not be expired)
        (10, 5, False),
        # Token expires now, no buffer
        (0, 0, True),
    ],
)
def test_is_token_expired(exp_offset, buffer_seconds, should_expire):
    exp = int(time.time()) + exp_offset
    token = _create_jwt_token({"exp": exp, "sub": "user-1"}, add_defaults=False)
    result = mlrun.auth.utils.is_token_expired(token, buffer_seconds=buffer_seconds)
    assert result is should_expire


def test_is_token_expired_missing_exp():
    token = _create_jwt_token({"sub": "user-1"}, add_defaults=False)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="Token is missing the 'exp'"
    ):
        mlrun.auth.utils.is_token_expired(token)


@pytest.mark.parametrize(
    "token_payload, expected_username",
    [
        # Token with preferred_username claim
        ({"preferred_username": "alice"}, "alice"),
        # Token without preferred_username claim
        ({}, None),
        # Token with empty preferred_username
        ({"preferred_username": ""}, ""),
    ],
)
def test_resolve_jwt_username(token_payload, expected_username):
    """Test extracting 'preferred_username' claim from JWT token."""
    jwt_token = _create_jwt_token(token_payload, add_defaults=True)
    result = mlrun.auth.utils.resolve_jwt_username(jwt_token, raise_on_error=False)
    assert result == expected_username


def test_resolve_jwt_username_invalid_token():
    """Test that resolve_jwt_username handles invalid tokens gracefully."""
    result = mlrun.auth.utils.resolve_jwt_username(
        "not-a-valid-jwt", raise_on_error=False
    )
    assert result is None
