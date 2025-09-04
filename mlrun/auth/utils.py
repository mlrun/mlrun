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
import typing

import yaml

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.helpers
from mlrun.config import config


def load_secret_tokens_from_file(
    token_file: typing.Optional[str] = None,
    raise_on_error: bool = True,
) -> list[dict]:
    """
    Load and parse secret tokens from a file.

    This function reads the token file and returns the raw list of token dictionaries.
    It does NOT validate the tokens.

    :param token_file: Path to the secret tokens file. If None, uses MLRun config.
    :param raise_on_error: Whether to raise exceptions on read/parse failure.
    :return: List of token dictionaries from 'secretTokens'.
    """
    token_file = token_file or config.auth_with_oauth_token.auth_token_file
    data = _read_secret_tokens_file(token_file, raise_on_error=raise_on_error)
    if not data:
        return []

    tokens_list = data.get("secretTokens")
    if not isinstance(tokens_list, list) or not tokens_list:
        mlrun.utils.helpers.raise_or_log_error(
            f"Invalid token file: 'secretTokens' must be a non-empty list in {token_file}",
            raise_on_error,
        )
        return []

    return tokens_list


def validate_secret_tokens(
    tokens_list: list[dict], token_file: str, raise_on_error: bool = True
) -> list[mlrun.common.schemas.SecretToken]:
    """
    Validate a list of token dictionaries and convert to SecretToken objects.

    Checks performed:
      - Each token has a non-empty 'name' and 'token'.
      - No duplicate token names.

    :param tokens_list: List of token dictionaries.
    :param token_file: Path to the file (used in error messages).
    :param raise_on_error: Whether to raise exceptions on invalid entries.
    :return: List of validated SecretToken objects.
    """
    tokens = []
    seen = set()

    for token in tokens_list:
        name = token.get("name")
        token_value = token.get("token")

        if not name or not token_value:
            mlrun.utils.helpers.raise_or_log_error(
                f"Invalid token entry in {token_file}: missing 'name' or 'token'",
                raise_on_error,
            )
            continue

        if name in seen:
            mlrun.utils.helpers.raise_or_log_error(
                f"Duplicate token name '{name}' found in {token_file}",
                raise_on_error,
            )
            continue

        seen.add(name)
        try:
            tokens.append(mlrun.common.schemas.SecretToken(**token))
        except Exception as exc:
            mlrun.utils.helpers.raise_or_log_error(
                f"Failed to create SecretToken from entry in {token_file}: {exc}",
                raise_on_error,
            )

    return tokens


def _read_secret_tokens_file(
    token_file: str, raise_on_error: bool = True
) -> typing.Optional[dict]:
    """
    Read and parse a secret tokens file as a dictionary.

    :param token_file: Path to the secret tokens file.
    :param raise_on_error: Whether to raise exceptions on failure.
    :return: Parsed file content as a dictionary, or None if an error occurs.
    """
    if not os.path.exists(token_file):
        mlrun.utils.helpers.raise_or_log_error(
            f"Token file not found at {token_file}", raise_on_error
        )
        return None

    try:
        with open(token_file) as token_file_io:
            return yaml.safe_load(token_file_io)
    except yaml.YAMLError as exc:
        mlrun.utils.helpers.raise_or_log_error(
            f"Failed to parse token file {token_file}: {exc}", raise_on_error
        )
        return None
