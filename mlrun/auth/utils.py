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
import time
import typing

import jwt
import yaml

import mlrun.common.constants
import mlrun.common.schemas
import mlrun.utils.helpers
from mlrun.config import config as mlconf

if typing.TYPE_CHECKING:
    import mlrun.db


class Claims:
    """
    JWT Claims constants.
    """

    SUBJECT = "sub"
    EXPIRATION = "exp"


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
    auth_user_id: str | None = None,
    raise_on_error: bool = True,
) -> list[mlrun.common.schemas.SecretToken]:
    """
    Load, validate, and translate secret tokens from a file into SecretToken objects.

    Steps performed:
      1. Load the secret tokens from the configured file.
      2. Validate each token for required fields and uniqueness.
      3. Translate validated token dictionaries into SecretToken objects.

    :param auth_user_id: The user ID to filter the tokens by.
    :param raise_on_error: Whether to raise exceptions or log warnings on failure
                           in any of the steps (loading, validation, translation).
    :return: List of SecretToken objects.
    :rtype: list[mlrun.common.schemas.SecretToken]
    """
    tokens_list = load_secret_tokens_from_file(raise_on_error=raise_on_error)
    validated_tokens = extract_and_validate_tokens_info(
        secret_tokens=[
            mlrun.common.schemas.SecretToken(
                name=token["name"],
                token=token["token"],
            )
            for token in tokens_list
        ],
        authenticated_id=auth_user_id,
        filter_by_authenticated_id=True,
    )
    secret_tokens = _translate_secret_tokens(
        validated_tokens, raise_on_error=raise_on_error
    )
    return secret_tokens


def extract_and_validate_tokens_info(
    secret_tokens: list[mlrun.common.schemas.SecretToken],
    authenticated_id: str,
    filter_by_authenticated_id: bool = False,
) -> dict[str, dict[str, typing.Any]]:
    """
    Extract and validate tokens info from a list of SecretToken objects.

    :param secret_tokens: List of SecretToken objects.
    :param authenticated_id: The authenticated user ID.
    :return: Dictionary of token info with the token name as the key and the token as the value.
    """
    token_values = {}
    for secret_token in secret_tokens:
        token_name = secret_token.name

        # Validate name is provided and not duplicate
        if secret_token.name and secret_token.name not in token_values:
            # The token is expected to be a refresh token which we cannot verify ourselves, we verify it separately
            # via orca when exchanging it for an access token. We decode it here without verification to extract its
            # claims.
            decoded_token = _decode_token_unverified(secret_token.token)

            # Validate token expiration existence
            if not decoded_token.get(Claims.EXPIRATION):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Offline token '{token_name}' is missing the 'exp' (expiration) claim"
                )
            # Validate token subject existence
            if not decoded_token.get(Claims.SUBJECT):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Offline token '{token_name}' is missing the 'sub' (subject) claim"
                )

            # Validate token belongs to the authenticated user
            token_sub = decoded_token.get(Claims.SUBJECT)
            if token_sub != authenticated_id:
                # just ignore the token as it doesn't belong to the authenticated user
                if filter_by_authenticated_id:
                    continue
                mlrun.utils.logger.warning(
                    "Offline token subject does not match the authenticated user",
                    token_name=token_name,
                    token_sub=token_sub,
                    user_id=authenticated_id,
                )
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Offline token '{token_name}' does not match the authenticated user ID. "
                    "Stored tokens can only belong to the authenticated user."
                )

            # Store token info
            token_values[secret_token.name] = {
                "token_exp": decoded_token.get(Claims.EXPIRATION),
                "token": secret_token.token,
            }
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid or duplicate token name '{secret_token.name}' found in request payload"
            )
    return token_values


def resolve_jwt_subject(
    token: str, raise_on_error: bool = True
) -> typing.Optional[str]:
    """
    Extract the 'sub' (subject/user ID) claim from a JWT token.

    The token is decoded without signature verification since it has already
    been verified earlier during the authentication process.

    :param token: The JWT token string.
    :param raise_on_error: Whether to raise an error or log a warning on failure.
    :return: The 'sub' claim value, or None if extraction fails.
    """
    try:
        # This method is used from the client side after receiving this token from the server, there's no need or
        # ability to verify its signature here.
        return _decode_token_unverified(token).get(Claims.SUBJECT)
    except jwt.PyJWTError as exc:
        mlrun.utils.helpers.raise_or_log_error(
            f"Failed to decode JWT token: {exc}", raise_on_error
        )
        return None


def is_token_expired(token: str, buffer_seconds: int = 0) -> bool:
    """
    Check if a JWT token is expired based on its 'exp' claim.

    :param token: The JWT token string.
    :param buffer_seconds: Number of seconds to subtract from the expiration time
    :return: True if the token is expired, False otherwise.
    """

    # This method is used for caching and/or extra validation purposes in addition to the main verification flow,
    # so we decode without signature verification here.
    decoded_token = _decode_token_unverified(token)
    expiration = decoded_token.get(Claims.EXPIRATION)
    if not expiration:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Token is missing the 'exp' (expiration) claim"
        )
    now = time.time()
    return now >= expiration - buffer_seconds


def _decode_token_unverified(token: str) -> dict:
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except jwt.DecodeError as exc:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Failed to decode offline token"
        ) from exc
    except Exception as exc:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Unexpected error decoding token"
        ) from exc


def _translate_secret_tokens(
    tokens_dict: dict[str, dict[str, typing.Any]], raise_on_error: bool = True
) -> list[mlrun.common.schemas.SecretToken]:
    """
    Translate a dictionary of validated token data into SecretToken objects.

    The dictionary is keyed by token name, with values containing token data
    (including the token string). If an entry fails to translate, behavior depends
    on ``raise_on_error``: raise an exception or log a warning.

    :param tokens_dict: Dictionary of validated token data, keyed by token name.
    :param raise_on_error: Whether to raise exceptions on translation errors.
    :return: List of SecretToken objects created from the input dictionary.
    :rtype: list[mlrun.common.schemas.SecretToken]
    """
    token_file = os.path.expanduser(mlconf.auth_with_oauth_token.token_file)
    tokens = []
    for token_name, token_data in tokens_dict.items():
        try:
            tokens.append(
                mlrun.common.schemas.SecretToken(
                    name=token_name,
                    token=token_data["token"],
                )
            )
        except Exception as exc:
            mlrun.utils.helpers.raise_or_log_error(
                f"Failed to create SecretToken from entry in {token_file}: {exc}",
                raise_on_error,
            )
    return tokens
