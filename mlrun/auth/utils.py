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
    data = read_secret_tokens_file(raise_on_error=raise_on_error)
    if not data:
        return None
    return parse_offline_token_data(data=data, raise_on_error=raise_on_error)


def read_secret_tokens_file(raise_on_error: bool = True) -> typing.Optional[dict]:
    """
    Read and parse the secret tokens file.

    This function attempts to read the token file specified in the configuration and parse its content as YAML.
    If the file does not exist or cannot be parsed, it either raises an error or logs a warning based on the
    `raise_on_error` parameter.

    Supports both ``.yaml`` and ``.yml`` extensions and will attempt to use the
    alternate extension if the file with the configured extension does not exist.

    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The parsed content of the token file as a dictionary, or None if an error occurs.
    """
    token_file = os.path.expanduser(config.auth_with_oauth_token.auth_token_file)

    # If the file doesn't exist, try the alternative extension
    if not os.path.exists(token_file):
        base, ext = os.path.splitext(token_file)
        if ext in [".yml", ".yaml"]:
            alt_ext = ".yaml" if ext == ".yml" else ".yml"
            alt_file = base + alt_ext
            if os.path.exists(alt_file):
                token_file = alt_file
            else:
                mlrun.utils.helpers.raise_or_log_error(
                    f"Token file not found at {token_file} or {alt_file}",
                    raise_on_error,
                )
                return None
        else:
            mlrun.utils.helpers.raise_or_log_error(
                f"Token file not found at {token_file}", raise_on_error
            )
            return None
    try:
        with open(token_file) as token_file_io:
            data = yaml.safe_load(token_file_io)
        if not data:
            mlrun.utils.helpers.raise_or_log_error(
                f"Token file {token_file} is empty or invalid",
                raise_on_error,
            )
            return None
        if not isinstance(data, dict):
            mlrun.utils.helpers.raise_or_log_error(
                f"Token file {token_file} must contain a YAML mapping (dictionary)",
                raise_on_error,
            )
            return None
        return data
    except yaml.YAMLError as exc:
        mlrun.utils.helpers.raise_or_log_error(
            f"Failed to parse token file {token_file}: {exc}", raise_on_error
        )
        return None


def parse_offline_token_data(
    data: dict, raise_on_error: bool = True
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
    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The resolved offline token, or None if resolution fails.
    """
    if not data:
        mlrun.utils.helpers.raise_or_log_error("Empty token data", raise_on_error)
        return None
    tokens = data.get("secretTokens")
    if not isinstance(tokens, list) or not tokens:
        mlrun.utils.helpers.raise_or_log_error(
            "Invalid token file: 'secretTokens' must be a non-empty list",
            raise_on_error,
        )
        return None

    name = config.auth_with_oauth_token.auth_token_name or "default"
    matches = [t for t in tokens if t.get("name") == name] or (
        [tokens[0]] if not config.auth_with_oauth_token.auth_token_name else []
    )

    if len(matches) != 1:
        mlrun.utils.helpers.raise_or_log_error(
            f"Failed to resolve a unique token. Found {len(matches)} entries for name '{name}'",
            raise_on_error,
        )
        return None

    token_value = matches[0].get("token")
    if not token_value:
        mlrun.utils.helpers.raise_or_log_error(
            "Resolved token entry missing 'token' field",
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
