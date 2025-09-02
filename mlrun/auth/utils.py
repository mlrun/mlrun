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

import mlrun.utils.helpers
from mlrun.config import config


def load_offline_token(raise_on_error=True) -> typing.Optional[str]:
    """
    Load the offline token from the environment variable or YAML file.

    The function first attempts to retrieve the offline token from the environment variable.
    If not found, it tries to load the token from a YAML file. If both methods fail, it either
    raises an error or logs a warning based on the `raise_on_error` parameter.

    :param raise_on_error: If True, raises an error when the offline token cannot be resolved.
                           If False, logs a warning instead.
    :return: The offline token if found, otherwise None.
    """
    if token_env := get_offline_token_from_env():
        return token_env
    return get_offline_token_from_file(raise_on_error=raise_on_error)


def get_offline_token_from_file(raise_on_error: bool = True) -> typing.Optional[str]:
    """
    Retrieve the offline token from a configured file.

    This function reads the token file specified in the configuration, parses its content,
    and extracts the offline token. If the file does not exist or cannot be parsed, it either
    raises an error or logs a warning based on the `raise_on_error` parameter.

    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The offline token if found, otherwise None.
    """
    token_file = config.auth_with_oauth_token.auth_token_file
    if not os.path.exists(token_file):
        mlrun.utils.helpers.raise_or_log_error(
            f"Token file not found at {token_file}", raise_on_error
        )
        return None

    try:
        with open(token_file) as token_file_io:
            data = yaml.safe_load(token_file_io)
    except yaml.YAMLError as exc:
        mlrun.utils.helpers.raise_or_log_error(
            f"Failed to parse token file {token_file}: {exc}", raise_on_error
        )
        return None

    return parse_offline_token_data(data, token_file, raise_on_error)


def parse_offline_token_data(
    data: dict, token_file: str, raise_on_error: bool = True
) -> typing.Optional[str]:
    """
    Extract the correct offline token entry from parsed YAML.

    Logic:
    1. Extract the `secretTokens` list from the parsed YAML file.
       - The list must be non-empty and contain objects.
       - If `secretTokens` is missing or invalid, resolution fails.

    2. Identify the target token entry using `mlrun.mlconf.auth_with_oauth_token.auth_token_name`:
       - If the value is set (non-empty):
         - Look for an entry where `name == <TOKEN_NAME>`.
         - If no match is found, resolution fails.
       - If the value is not set (empty string):
         - Look for an entry named "default".
         - If not found, fall back to the first token in the list.
         - If no entries exist, resolution fails.

    3. Validate the matched entry:
       - Ensure the `token` field exists and is a valid, non-empty string.
       - If valid, use the token as the resolved Offline Token.

    4. If any of the above steps fail, raise a detailed configuration error or log a warning.

    :param data: The parsed YAML data.
    :param token_file: The path to the token file.
    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The resolved offline token, or None if resolution fails.
    """
    tokens = data.get("secretTokens")
    if not isinstance(tokens, list) or not tokens:
        mlrun.utils.helpers.raise_or_log_error(
            f"Invalid token file: 'secretTokens' must be a non-empty list in {token_file}",
            raise_on_error,
        )
        return None

    name = config.auth_with_oauth_token.auth_token_name or "default"
    matches = [t for t in tokens if t.get("name") == name] or (
        [tokens[0]] if not config.auth_with_oauth_token.auth_token_name else []
    )

    if len(matches) != 1:
        mlrun.utils.helpers.raise_or_log_error(
            f"Failed to resolve a unique token. Found {len(matches)} entries for name '{name}' in {token_file}",
            raise_on_error,
        )
        return None

    token_value = matches[0].get("token")
    if not token_value:
        mlrun.utils.helpers.raise_or_log_error(
            f"Resolved token entry missing 'token' field in {token_file}",
            raise_on_error,
        )
        return None

    return token_value


def get_offline_token_from_env() -> typing.Optional[str]:
    """
    Retrieve the offline token from the environment variable.

    This function checks the environment for the `MLRUN_AUTH_OFFLINE_TOKEN` variable
    and returns its value if set.

    :return: The offline token if found in the environment, otherwise None.
    """
    return mlrun.secrets.get_secret_or_env("MLRUN_AUTH_OFFLINE_TOKEN")
