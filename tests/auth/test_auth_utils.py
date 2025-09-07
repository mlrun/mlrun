# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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

import pytest

import mlrun.common.schemas
import mlrun.errors
from mlrun.auth import utils as auth_utils


def _write_file(tmp_path, name: str, content: str) -> str:
    """Helper to write YAML content to a file under tmp_path and return the path."""
    file_path = tmp_path / name
    file_path.write_text(content)
    return str(file_path)


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

    # Use the new combined utility function
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
        auth_utils._validate_secret_tokens(tokens_list, path)


def test_read_secret_tokens_file_non_existent(tmp_path):
    """Test reading a file that does not exist."""
    file_path = tmp_path / "does_not_exist.yml"

    # Should return None if raise_on_error=False
    result = auth_utils._read_secret_tokens_file(str(file_path), raise_on_error=False)
    assert result is None

    # Should raise MLRunRuntimeError if raise_on_error=True
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        auth_utils._read_secret_tokens_file(str(file_path), raise_on_error=True)
