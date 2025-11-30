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

import textwrap
from unittest.mock import patch

import pytest
import yaml

import mlrun.auth.utils
import mlrun.common.schemas
import mlrun.errors
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
            [{"name": "default", "token": "file-token"}],
            None,
            "file-token",
        ),
        # 2. Valid token with custom name
        (
            [{"name": "custom", "token": "custom-token"}],
            "custom",
            "custom-token",
        ),
        # # 3. secretTokens not a list
        ("not-a-list", None, None),
        # # 4. secretTokens empty list
        ([], None, None),
        # 5. Multiple matching tokens
        (
            [
                {"name": "default", "token": "t1"},
                {"name": "default", "token": "t2"},
            ],
            None,
            None,
        ),
        # 6. Token entry missing 'token' field
        ([{"name": "default"}], None, None),
        # 7. Empty default token name, no default, use 1st token
        (
            [
                {"name": "token1", "token": "file-token1"},
                {"name": "token2", "token": "file-token2"},
            ],
            None,
            "file-token1",
        ),
    ],
)
def test_parse_offline_token_data_cases(data, token_name, expected_token, monkeypatch):
    monkeypatch.setattr(
        "mlrun.config.config.auth_with_oauth_token.token_name", token_name
    )
    # Suppress raising errors, we just check return value
    token = mlrun.auth.utils.parse_offline_token_data(data, raise_on_error=False)
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
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", str(fake_file))

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
        token = mlrun.auth.utils.get_offline_token_from_file(
            raise_on_error=raise_on_error
        )
        assert token == expected_token


@pytest.mark.parametrize(
    "content,expected_count",
    [
        (
            textwrap.dedent("""\
            secretTokens:
              - name: token1
                token: abc123
        """),
            1,
        ),
        (
            textwrap.dedent("""\
            secretTokens:
              - name: token1
                token: abc123
              - name: token2
                token: def456
        """),
            2,
        ),
    ],
)
def test_load_and_prepare_secret_tokens_valid(
    tmp_path, content, expected_count, monkeypatch
):
    path = _write_file(tmp_path, "tokens.yml", content)
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", path)

    secret_tokens = mlrun.auth.utils.load_and_prepare_secret_tokens()
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
    "content",
    [
        textwrap.dedent("""\
            secretTokens:
              - token: abc123
        """),
        textwrap.dedent("""\
            secretTokens:
              - name: dup
                token: abc123
              - name: dup
                token: def456
        """),
        textwrap.dedent("""\
            secretTokens:
              - name: missing_token
        """),
    ],
)
def test_validate_secret_tokens_invalid_entries(tmp_path, content, monkeypatch):
    path = _write_file(tmp_path, "tokens.yml", content)
    monkeypatch.setattr(config.auth_with_oauth_token, "token_file", path)
    tokens_list = mlrun.auth.utils.load_secret_tokens_from_file(raise_on_error=False)
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        mlrun.auth.utils.validate_secret_tokens(tokens_list)


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
