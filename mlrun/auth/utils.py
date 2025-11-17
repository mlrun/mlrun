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
import mlrun.utils.helpers
from mlrun.config import config as mlconf

if typing.TYPE_CHECKING:
    import mlrun.db


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
    tokens = load_secret_tokens_from_file(raise_on_error=raise_on_error)
    if not tokens:
        return None
    return parse_offline_token_data(tokens=tokens, raise_on_error=raise_on_error)


def load_secret_tokens_from_file(
    raise_on_error: bool = True,
) -> list[dict]:
    """
    Load and parse secret tokens from a configured file.

    This function reads the secret tokens file (specified in
    ``mlrun.mlconf.auth_with_oauth_token.token_file``) and returns the raw list
    of token dictionaries under the ``secretTokens`` key. It does NOT validate
    the tokens.

    If the file is missing, empty, or malformed, the behavior depends on
    ``raise_on_error``. In such cases, the function will either raise/log an
    error and return an empty list.

    :param raise_on_error: Whether to raise exceptions on read/parse failure.
    :return: List of token dictionaries from ``secretTokens``.
             Returns an empty list if parsing fails or no tokens exist.
    :rtype: list[dict[str, Any]]
    """
    token_file = os.path.expanduser(mlconf.auth_with_oauth_token.token_file)
    data = read_secret_tokens_file(raise_on_error=raise_on_error)
    if not data:
        mlrun.utils.helpers.raise_or_log_error(
            f"Token file is empty or could not be parsed: {token_file}",
            raise_on_error,
        )
        return []

    tokens_list = data.get("secretTokens")
    if not isinstance(tokens_list, list) or not tokens_list:
        mlrun.utils.helpers.raise_or_log_error(
            f"Invalid token file: 'secretTokens' must be a non-empty list in {token_file}",
            raise_on_error,
        )
        return []

    return tokens_list


def read_secret_tokens_file(
    raise_on_error: bool = True,
) -> typing.Optional[dict[str, typing.Any]]:
    """
    Read and parse the secret tokens file.

    This function attempts to read the token file specified in the configuration and parse its content as YAML.
    If the file does not exist or cannot be parsed, it either raises an error or logs a warning based on the
    `raise_on_error` parameter.

    - The configured path may use ``~`` to represent the user’s home directory, which
      will be expanded automatically.

    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The parsed content of the token file as a dictionary, or None if an error occurs.
    """
    token_file = os.path.expanduser(mlconf.auth_with_oauth_token.token_file)

    if not os.path.exists(token_file):
        mlrun.utils.helpers.raise_or_log_error(
            f"Configured token file not found: {token_file}", raise_on_error
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
    tokens: list[dict[str, typing.Any]], raise_on_error: bool = True
) -> typing.Optional[str]:
    """
    Extract the correct offline token entry from the parsed tokens list.

    Logic:
    1. Identify the target token entry using `mlrun.mlconf.auth_with_oauth_token.token_name`:
       - If the value is set (non-empty):
         - Look for an entry where `name == <TOKEN_NAME>`.
         - If no match is found, resolution fails.
       - If the value is not set (empty string):
         - Look for an entry named "default".
         - If not found, fall back to the first token in the list.
         - If no entries exist, resolution fails.
    2. Validate the matched entry:
       - Ensure the `token` field exists and is a valid, non-empty string.
       - If valid, use the token as the resolved Offline Token.
    3. If any of the above steps fail, raise a detailed configuration error or log a warning.


    :param tokens: List of token dictionaries loaded from the YAML file.
    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The resolved offline token, or None if resolution fails.
    """
    if not isinstance(tokens, list) or not tokens:
        mlrun.utils.helpers.raise_or_log_error(
            "Invalid token file: 'secretTokens' must be a non-empty list",
            raise_on_error,
        )
        return None

    name = mlconf.auth_with_oauth_token.token_name or "default"
    matches = [t for t in tokens if t.get("name") == name] or (
        [tokens[0]] if not mlconf.auth_with_oauth_token.token_name else []
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


def load_and_prepare_secret_tokens(
    raise_on_error: bool = True,
) -> list[mlrun.common.schemas.SecretToken]:
    """
    Load, validate, and translate secret tokens from a file into SecretToken objects.

    Steps performed:
      1. Load the secret tokens from the configured file.
      2. Validate each token for required fields and uniqueness.
      3. Translate validated token dictionaries into SecretToken objects.

    :param raise_on_error: Whether to raise exceptions or log warnings on failure
                           in any of the steps (loading, validation, translation).
    :return: List of SecretToken objects.
    :rtype: list[mlrun.common.schemas.SecretToken]
    """
    tokens_list = load_secret_tokens_from_file(raise_on_error=raise_on_error)
    validated_tokens = validate_secret_tokens(
        tokens_list, raise_on_error=raise_on_error
    )
    secret_tokens = translate_secret_tokens(
        validated_tokens, raise_on_error=raise_on_error
    )
    return secret_tokens


def validate_secret_tokens(
    tokens_list: list[dict[str, typing.Any]], raise_on_error: bool = True
) -> list[dict[str, typing.Any]]:
    """
    Validate a list of token dictionaries.

    Checks performed:
      - Each token has a non-empty 'name' and 'token'.
        If raise_on_error=False, invalid entries are ignored.
      - No duplicate token names.
        If raise_on_error=False, duplicates are ignored.

    :param tokens_list: List of token dictionaries to validate.
    :param raise_on_error: Whether to raise exceptions on invalid entries.
    :return: List of validated token dictionaries.
    :rtype: list[dict[str, Any]]
    """
    valid_tokens = []
    seen = set()

    token_file = os.path.expanduser(mlconf.auth_with_oauth_token.token_file)
    for token in tokens_list:
        name = token.get("name")
        token_value = token.get("token")

        if not name or not isinstance(token_value, str) or not token_value.strip():
            # Invalid entry
            mlrun.utils.helpers.raise_or_log_error(
                f"Invalid token entry in {token_file}: missing or empty 'name' or 'token'",
                raise_on_error,
            )
            continue

        if name in seen:
            # Duplicate entry
            mlrun.utils.helpers.raise_or_log_error(
                f"Duplicate token name '{name}' found in {token_file}",
                raise_on_error,
            )
            continue

        seen.add(name)
        valid_tokens.append(token)

    return valid_tokens


def translate_secret_tokens(
    tokens_list: list[dict[str, typing.Any]], raise_on_error: bool = True
) -> list[mlrun.common.schemas.SecretToken]:
    """
    Translate a list of validated token dictionaries into SecretToken objects.

    Each dictionary in the list must be validated and contain the required fields
    for SecretToken creation. If an entry fails to translate, behavior depends on
    ``raise_on_error``: raise an exception or log a warning.

    :param tokens_list: List of validated token dictionaries.
    :param raise_on_error: Whether to raise exceptions on translation errors.
    :return: List of SecretToken objects created from the input dictionaries.
    :rtype: list[mlrun.common.schemas.SecretToken]
    """
    token_file = os.path.expanduser(mlconf.auth_with_oauth_token.token_file)
    tokens = []
    for token in tokens_list:
        try:
            tokens.append(mlrun.common.schemas.SecretToken(**token))
        except Exception as exc:
            mlrun.utils.helpers.raise_or_log_error(
                f"Failed to create SecretToken from entry in {token_file}: {exc}",
                raise_on_error,
            )
    return tokens


def enrich_auth_env(
    env: dict,
    db: "mlrun.db.RunDBInterface",
    auth_info: mlrun.common.schemas.AuthInfo = None,
):
    """
    Enrich the given environment dictionary with authentication information.

    This function adds authentication-related environment variables to the provided
    environment dictionary based on the given AuthInfo object.

    :param env: The environment dictionary to enrich.
    :param db: The RunDBInterface instance to retrieve secret tokens.
    :param auth_info: The AuthInfo object containing authentication details.
    """

    # TODO: Remove this once we implement secret token mounting in jobs (ML-11292)
    _default_token_name = "default"

    if mlrun.mlconf.is_iguazio_v4_mode():
        if auth_info and auth_info.username:
            secret = db.get_secret_token(
                token_name=_default_token_name,
                username=auth_info.username,
            )
            env["MLRUN_AUTH_OFFLINE_TOKEN"] = secret.token

        env["MLRUN_AUTH_WITH_OAUTH_TOKEN__ENABLED"] = "true"
        env["MLRUN_AUTH_TOKEN_ENDPOINT"] = (
            mlrun.mlconf.iguazio_api_url + "/api/v1/refresh-access-token"
        )
        env["MLRUN_HTTPDB__HTTP__VERIFY"] = str(
            mlrun.mlconf.iguazio_api_ssl_verify
        ).lower()
