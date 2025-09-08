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

import os
import textwrap
from unittest.mock import patch

import pytest
import yaml

import mlrun.auth.utils
import mlrun.common.schemas
import mlrun.errors
from mlrun.auth import utils as auth_utils
from mlrun.config import config


def test_get_offline_token_from_env(monkeypatch):
    monkeypatch.setenv("MLRUN_AUTH_OFFLINE_TOKEN", "env-token")
    token = mlrun.auth.utils.get_offline_token_from_env()
    assert token == "env-token"
    monkeypatch.delenv("MLRUN_AUTH_OFFLINE_TOKEN", raising=False)
    assert mlrun.auth.utils.get_offline_token_from_env() is None


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
    # Suppress raising errors, we just check return value
    token = mlrun.auth.utils.parse_offline_token_data(data, raise_on_error=False)
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

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        mlrun.auth.utils.parse_offline_token_data(data, raise_on_error=True)


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
    with (
        patch.object(
            mlrun.auth.utils, "get_offline_token_from_env", return_value=env_token
        ),
        patch.object(
            mlrun.auth.utils, "get_offline_token_from_file", return_value=file_token
        ),
    ):
        token = mlrun.auth.utils.load_offline_token(raise_on_error=False)
        assert token == expected


def test_token_file_not_exists(monkeypatch):
    fake_file = "no_such_file.yaml"
    monkeypatch.setattr(config.auth_with_oauth_token, "auth_token_file", str(fake_file))

    result = mlrun.auth.utils.get_offline_token_from_file(raise_on_error=False)
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
        with open(token_file, "w") as f:
            yaml.safe_dump(file_content, f)

    # Monkeypatch config to point to temp file
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.auth_token_file", str(token_file)
    )

    if expected_token is None and raise_on_error:
        # Expect MLRunRuntimeError
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            mlrun.auth.utils.get_offline_token_from_file(raise_on_error=True)
    else:
        token = mlrun.auth.utils.get_offline_token_from_file(
            raise_on_error=raise_on_error
        )
        assert token == expected_token


@pytest.mark.parametrize(
    "content,expected_count",
    [
        (
            textwrap.dedent(
                """
                secretTokens:
                  - name: token1
                    token: abc123
                """
            ),
            1,
        ),
        (
            textwrap.dedent(
                """
                secretTokens:
                  - name: token1
                    token: abc123
                  - name: token2
                    token: def456
                """
            ),
            2,
        ),
    ],
)
def test_load_and_prepare_secret_tokens_valid(tmp_path, content, expected_count):
    """Test loading and validating valid secret tokens from file."""
    path = _write_file(tmp_path, "tokens.yml", content)

    secret_tokens = auth_utils.load_and_prepare_secret_tokens(path)
    assert isinstance(secret_tokens, list)
    assert len(secret_tokens) == expected_count
    assert all(isinstance(t, mlrun.common.schemas.SecretToken) for t in secret_tokens)


@pytest.mark.parametrize(
    "content",
    [
        # Missing secretTokens field
        textwrap.dedent(
            """
            notSecretTokens:
              - name: token1
                token: abc123
            """
        ),
        # Empty secretTokens list
        textwrap.dedent(
            """
            secretTokens: []
            """
        ),
        # Wrong type (dict instead of list)
        textwrap.dedent(
            """
            secretTokens:
              token1: abc123
            """
        ),
    ],
)
def test_load_secret_tokens_from_file_invalid(tmp_path, content):
    """
    Test that loading invalid secret token files raises MLRunRuntimeError.
    """
    path = _write_file(tmp_path, "tokens.yml", content)
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        auth_utils.load_secret_tokens_from_file(path)


@pytest.mark.parametrize(
    "content",
    [
        # Missing name field
        textwrap.dedent(
            """
            secretTokens:
              - token: abc123
            """
        ),
        # Duplicate name
        textwrap.dedent(
            """
            secretTokens:
              - name: dup
                token: abc123
              - name: dup
                token: def456
            """
        ),
        # Missing token field
        textwrap.dedent(
            """
            secretTokens:
              - name: missing_token
            """
        ),
    ],
)
def test_validate_secret_tokens_invalid_entries(tmp_path, content):
    """
    Test that validate_secret_tokens raises MLRunRuntimeError for invalid token entries:
    - Missing 'name' or 'token' field
    - Duplicate token names
    """
    path = _write_file(tmp_path, "tokens.yml", content)

    # Load tokens without raising for file-level issues
    tokens_list = auth_utils.load_secret_tokens_from_file(path, raise_on_error=False)

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        auth_utils.validate_secret_tokens(tokens_list, path)


def test_read_secret_tokens_file_non_existent(tmp_path):
    """Test reading a file that does not exist."""
    file_path = tmp_path / "does_not_exist.yml"

    # Should return None if raise_on_error=False
    result = auth_utils.read_secret_tokens_file(str(file_path), raise_on_error=False)
    assert result is None

    # Should raise MLRunRuntimeError if raise_on_error=True
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        auth_utils.read_secret_tokens_file(str(file_path), raise_on_error=True)


def test_read_secret_tokens_file_alternative_extension(tmp_path):
    """Test that the function correctly reads a file with alternative extension fallback."""
    yml_content = """
    secretTokens:
      - name: token1
        token: abc123
    """
    yaml_content = """
    secretTokens:
      - name: token2
        token: def456
    """

    # Write both files
    yml_path = _write_file(tmp_path, "tokens.yml", yml_content)
    _write_file(tmp_path, "tokens.yaml", yaml_content)

    # Case 1: Read existing .yml
    result = auth_utils.read_secret_tokens_file(yml_path)
    assert result["secretTokens"][0]["name"] == "token1"

    # Remove .yml to force fallback to .yaml
    os.remove(yml_path)

    # Case 2: Read non-existent .yml, should fallback to .yaml
    result = auth_utils.read_secret_tokens_file(yml_path)
    assert result["secretTokens"][0]["name"] == "token2"


def _write_file(tmp_path, name: str, content: str) -> str:
    """Helper to write YAML content to a file under tmp_path and return the path."""
    file_path = tmp_path / name
    file_path.write_text(content)
    return str(file_path)
