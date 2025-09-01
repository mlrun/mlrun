# Copyright 2024 Iguazio
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
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import jwt
import requests
import yaml

import mlrun.errors
import mlrun.secrets
import mlrun.utils.helpers
from mlrun.config import config
from mlrun.utils import logger


class TokenProvider(ABC):
    @abstractmethod
    def get_token(self):
        pass

    @abstractmethod
    def is_iguazio_session(self):
        pass


class StaticTokenProvider(TokenProvider):
    def __init__(self, token: str):
        self.token = token

    def get_token(self):
        return self.token

    def is_iguazio_session(self):
        return mlrun.platforms.iguazio.is_iguazio_session(self.token)


class OAuthClientIDTokenProvider(TokenProvider):
    def __init__(
        self, token_endpoint: str, client_id: str, client_secret: str, timeout=5
    ):
        if not token_endpoint or not client_id or not client_secret:
            raise mlrun.errors.MLRunValueError(
                "Invalid client_id configuration for authentication. Must provide token endpoint, client-id and secret"
            )
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout

        # Since we're only issuing POST requests, which are actually a disguised GET, then it's ok to allow retries
        # on them.
        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_post=True,
            verbose=True,
        )

        self._cleanup()
        self._refresh_token_if_needed()

    def get_token(self):
        self._refresh_token_if_needed()
        return self.token

    def is_iguazio_session(self):
        return False

    def _cleanup(self):
        self.token = self.token_expiry_time = self.token_refresh_time = None

    def _refresh_token_if_needed(self):
        now = datetime.now()
        if self.token:
            if self.token_refresh_time and now <= self.token_refresh_time:
                return self.token

            # We only cleanup if token was really expired - even if we fail in refreshing the token, we can still
            # use the existing one given that it's not expired.
            if now >= self.token_expiry_time:
                self._cleanup()

        self._issue_token_request()
        return self.token

    def _issue_token_request(self, raise_on_error=False):
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            request_body = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            response = self._session.request(
                "POST",
                self.token_endpoint,
                timeout=self.timeout,
                headers=headers,
                data=request_body,
            )
        except requests.RequestException as exc:
            error = f"Retrieving token failed: {mlrun.errors.err_to_str(exc)}"
            if raise_on_error:
                raise mlrun.errors.MLRunRuntimeError(error) from exc
            else:
                logger.warning(error)
                return

        if not response.ok:
            error = "No error available"
            if response.content:
                try:
                    data = response.json()
                    error = data.get("error")
                except Exception:
                    pass
            logger.warning(
                "Retrieving token failed", status=response.status_code, error=error
            )
            if raise_on_error:
                mlrun.errors.raise_for_status(response)
            return

        self._parse_response(response.json())

    def _parse_response(self, data: dict):
        # Response is described in https://datatracker.ietf.org/doc/html/rfc6749#section-4.4.3
        # According to spec, there isn't a refresh token - just the access token and its expiry time (in seconds).
        self.token = data.get("access_token")
        expires_in = data.get("expires_in")
        if not self.token or not expires_in:
            token_str = "****" if self.token else "missing"
            logger.warning(
                "Failed to parse token response", token=token_str, expires_in=expires_in
            )
            return

        now = datetime.now()
        self.token_expiry_time = now + timedelta(seconds=expires_in)
        self.token_refresh_time = now + timedelta(seconds=expires_in / 2)
        logger.info(
            "Successfully retrieved client-id token",
            expires_in=expires_in,
            expiry=str(self.token_expiry_time),
            refresh=str(self.token_refresh_time),
        )


class IGTokenProvider(TokenProvider):
    def __init__(self, token_endpoint: str):
        if not token_endpoint:
            raise mlrun.errors.MLRunValueError(
                "No token endpoint provided, cannot initialize token provider"
            )
        self.token_endpoint = token_endpoint
        self._access_token = None
        self._token_total_lifetime = 0
        self._token_expiry_time = None

        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_post=True,
            verbose=True,
        )

        self._refresh_access_token_if_needed(raise_on_error=True)

    def get_token(self):
        self._refresh_access_token_if_needed()
        return self._access_token

    def is_iguazio_session(self):
        return False

    def _refresh_access_token_if_needed(self, raise_on_error=False):
        # Check if there is an existing access token and if it is valid
        if self._access_token and self._is_access_token_valid():
            return

        # Use the offline token to fetch a new access token
        self._fetch_access_token(raise_on_error=raise_on_error)

    def _is_access_token_valid(self) -> bool:
        """
        Check if the current access token is valid and has sufficient lifetime remaining.

        :return: True if the token is valid and has more than the configured threshold of its lifetime remaining.
        """
        if (
            not self._access_token
            or self._token_total_lifetime <= 0
            or not self._token_expiry_time
        ):
            return False

        now = datetime.now()
        remaining_lifetime = (self._token_expiry_time - now).total_seconds()

        return (
            remaining_lifetime / self._token_total_lifetime
            > config.auth_with_oauth_token.refresh_threshold
        )

    @staticmethod
    def get_token_lifetime_and_expiry(
        token: str,
    ) -> tuple[int, typing.Optional[datetime]]:
        """
        Calculate the total lifetime and expiration time of the token.

        :param token: The access token to decode.
        :return: A tuple containing the total lifetime of the token in seconds and its expiration time as a datetime.
        """
        if not token:
            return 0, None
        try:
            decoded_token = jwt.decode(token, options={"verify_signature": False})
            exp_timestamp = decoded_token.get("exp")
            iat_timestamp = decoded_token.get("iat")
            if exp_timestamp and iat_timestamp:
                return exp_timestamp - iat_timestamp, datetime.fromtimestamp(
                    exp_timestamp
                )
        except jwt.PyJWTError as exc:
            logger.warning(
                "Failed to decode access token",
                error=str(exc),
            )
        return 0, None

    def _fetch_access_token(self, raise_on_error=False):
        """
        Fetch a new access token using the offline token.
        """
        offline_token = self._load_offline_token(raise_on_error=raise_on_error)
        if not offline_token:
            # Error already handled in `_load_offline_token`
            return

        try:
            headers = {"Content-Type": "application/json"}
            request_body = {"refreshToken": offline_token}
            response = self._session.request(
                "POST",
                self.token_endpoint,
                timeout=config.auth_with_oauth_token.request_timeout,
                headers=headers,
                json=request_body,
            )
            if not response.ok and raise_on_error:
                response.raise_for_status()
                return
            response_data = response.json()
            self._parse_access_token_response(response_data)
        except requests.RequestException as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to fetch access token: {mlrun.errors.err_to_str(exc)}"
            ) from exc

    def _parse_access_token_response(self, response_data: dict, raise_on_error=False):
        """
        Parse the response from the access token endpoint.

        :param response_data: The JSON response from the access token endpoint.
        """
        spec = response_data.get("spec", {})
        access_token = spec.get("accessToken")

        if not access_token:
            mlrun.utils.helpers.raise_or_log_error(
                "Access token is missing in the response from the token endpoint",
                raise_on_error,
            )
            return

        self._access_token = access_token
        self._token_total_lifetime, self._token_expiry_time = (
            self.get_token_lifetime_and_expiry(access_token)
        )

    def _load_offline_token(self, raise_on_error=True) -> typing.Optional[str]:
        """
        Load the offline token from the environment variable or YAML file.

        :param raise_on_error: If True, raises an error when the offline token cannot be resolved.
                               If False, logs a warning instead.
        :return: The offline token if found, otherwise None.
        """
        if token_env := self._get_offline_token_from_env():
            return token_env
        return self._get_offline_token_from_file(raise_on_error)

    def _get_offline_token_from_file(
        self, raise_on_error: bool = True
    ) -> typing.Optional[str]:
        """Try resolving offline token from configured file."""
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

        return self._parse_offline_token_data(data, token_file, raise_on_error)

    @staticmethod
    def _parse_offline_token_data(
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

    @staticmethod
    def _get_offline_token_from_env() -> typing.Optional[str]:
        """Try resolving offline token from environment variable."""
        return mlrun.secrets.get_secret_or_env("MLRUN_AUTH_OFFLINE_TOKEN")
