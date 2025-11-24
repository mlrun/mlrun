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

import typing
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import jwt
import requests

import mlrun.auth.utils
import mlrun.errors
import mlrun.secrets
import mlrun.utils.helpers
from mlrun.config import config as mlconf
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


class DynamicTokenProvider(TokenProvider):
    """
    A token provider that dynamically fetches and refreshes tokens from a token endpoint.

    This class handles token retrieval and automatic refresh when the token is expired or about to expire.
    It uses a session with retry capabilities for robust communication with the token endpoint.

    :param token_endpoint: The URL of the token endpoint.
    :param timeout: The timeout for token requests, in seconds.
    """

    def __init__(self, token_endpoint: str, timeout=5, max_retries=0):
        if not token_endpoint:
            raise mlrun.errors.MLRunValueError(
                "No token endpoint provided, cannot initialize token provider"
            )
        self._token = None
        self._token_endpoint = token_endpoint
        self._timeout = timeout
        self._max_retries = max_retries

        # Since we're only issuing POST requests, which are actually a disguised GET, then it's ok to allow retries
        # on them.
        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_post=True,
            verbose=True,
        )
        self._cleanup()
        self._refresh_token_if_needed()

    def get_token(self):
        """
        Retrieve the current access token, refreshing it if necessary.

        :return: The current access token.
        """
        self._refresh_token_if_needed()
        return self._token

    def is_iguazio_session(self):
        return False

    def fetch_token(self):
        mlrun.utils.helpers.run_with_retry(
            retry_count=self._max_retries,
            func=self._fetch_token,
        )

    def _fetch_token(self):
        """
        Fetch a new access token from the token endpoint.

        This method builds the token request, sends it to the token endpoint, and parses the response.
        If the request fails, it either raises an error or logs a warning based on the `raise_on_error` parameter.
        """
        request_body, headers, body_type = self._build_token_request(
            raise_on_error=True
        )

        try:
            request_kwargs = {
                "method": "POST",
                "url": self._token_endpoint,
                "timeout": self._timeout,
                "headers": headers,
                "verify": mlconf.httpdb.http.verify,
            }
            if body_type == "json":
                request_kwargs["json"] = request_body
            else:
                request_kwargs["data"] = request_body

            response = self._session.request(**request_kwargs)
        except requests.RequestException as exc:
            error = f"Retrieving token failed: {mlrun.errors.err_to_str(exc)}"
            raise mlrun.errors.MLRunRuntimeError(error) from exc

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
            mlrun.errors.raise_for_status(response)

        self._parse_response(response.json())

    def _refresh_token_if_needed(self):
        """
        Refresh the access token if it is expired or about to expire.

        :return: The refreshed access token.
        """
        raise_on_error = True

        # Check if there is an existing access token and if it is within the refresh threshold
        if self._token and self._is_token_within_refresh_threshold(
            cleanup_if_expired=True
        ):
            return self._token

        try:
            self.fetch_token()
        except Exception as exc:
            raise_on_error = False
            # Token fetch failed and there is no existing token - cannot proceed
            if not self._token:
                raise mlrun.errors.MLRunRuntimeError(
                    "Failed to fetch a valid access token. Authentication procedure stopped."
                ) from exc

        finally:
            self._post_fetch_hook(raise_on_error)

        return self._token

    @abstractmethod
    def _post_fetch_hook(self, raise_on_error=True):
        """
        A hook that is called after fetching a new token.
        Can be used to perform additional actions, such as logging or updating state.
        """
        pass

    @abstractmethod
    def _is_token_within_refresh_threshold(self, cleanup_if_expired=True) -> bool:
        """
        Check if the current access token is valid.

        :param cleanup_if_expired: Whether to clean up the token if it is expired.
        :return: True if the token is valid, False otherwise.
        """
        pass

    @abstractmethod
    def _cleanup(self):
        """
        Clean up the token and related metadata.
        """
        pass

    @abstractmethod
    def _build_token_request(self, raise_on_error=False):
        """
        Build the request body and headers for the token request.

        :param raise_on_error: Whether to raise an error if the request cannot be built.
        :return: A tuple containing the request body and headers.
        """
        pass

    @abstractmethod
    def _parse_response(self, data: dict):
        """
        Parse the response from the token endpoint.

        :param data: The JSON response data from the token endpoint.
        """
        pass


class OAuthClientIDTokenProvider(DynamicTokenProvider):
    def __init__(
        self, token_endpoint: str, client_id: str, client_secret: str, timeout=5
    ):
        if not token_endpoint or not client_id or not client_secret:
            raise mlrun.errors.MLRunValueError(
                "Invalid client_id configuration for authentication. Must provide token endpoint, client-id and secret"
            )
        # should be set before calling the parent constructor
        self._client_id = client_id
        self._client_secret = client_secret
        super().__init__(token_endpoint=token_endpoint, timeout=timeout)

    def _cleanup(self):
        self._token = self.token_expiry_time = self.token_refresh_time = None

    def _is_token_within_refresh_threshold(self, cleanup_if_expired=True) -> bool:
        """
        Check if the current access token is valid.

        :param cleanup_if_expired: Whether to clean up the token if it is expired.
        :return: True if the token is valid, False otherwise.
        """
        if not self._token or not self.token_expiry_time:
            return False

        now = datetime.now()

        if now <= self.token_refresh_time:
            return True

        if now < self.token_expiry_time:
            # past refresh time but not expired yet → not valid
            return False

        # expired
        if cleanup_if_expired:
            # We only cleanup if token was really expired - even if we fail in refreshing the token, we can still
            # use the existing one given that it's not expired.
            self._cleanup()
        return False

    def _build_token_request(self, raise_on_error=False):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        request_body = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        return request_body, headers, "data"

    def _parse_response(self, data: dict):
        # Response is described in https://datatracker.ietf.org/doc/html/rfc6749#section-4.4.3
        # According to spec, there isn't a refresh token - just the access token and its expiry time (in seconds).
        self._token = data.get("access_token")
        expires_in = data.get("expires_in")
        if not self._token or not expires_in:
            token_str = "****" if self._token else "missing"
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

    def _post_fetch_hook(self, raise_on_error=True):
        """
        A hook that is called after fetching a new token.
        Can be used to perform additional actions, such as logging or updating state.
        """
        pass


class IGTokenProvider(DynamicTokenProvider):
    """
    A token provider for Iguazio that uses a refresh token to fetch access tokens.

    This class implements the Iguazio-specific token refresh flow to retrieve access tokens
    from a token endpoint.

    :param token_endpoint: The URL of the token endpoint.
    :param timeout: The timeout for token requests, in seconds.
    """

    def __init__(self, token_endpoint: str, timeout=5):
        super().__init__(token_endpoint=token_endpoint, timeout=timeout, max_retries=2)

    def _cleanup(self):
        self._token = None
        self._token_total_lifetime = 0
        self._token_expiry_time = None

    def _is_token_within_refresh_threshold(self, cleanup_if_expired=True) -> bool:
        """
        Check if the current access token is valid and has sufficient lifetime remaining.

        :param cleanup_if_expired: Whether to clean up the token if it is expired.
        :return: True if the token is valid, False otherwise.
        """
        if (
            not self._token
            or self._token_total_lifetime <= 0
            or not self._token_expiry_time
        ):
            return False

        now = datetime.now()
        remaining_lifetime = (self._token_expiry_time - now).total_seconds()
        if remaining_lifetime <= 0 and cleanup_if_expired:
            self._cleanup()
            return False

        return (
            self._token_total_lifetime - remaining_lifetime
            < self._token_total_lifetime
            * mlconf.auth_with_oauth_token.refresh_threshold
        )

    def _build_token_request(self, raise_on_error=False):
        """
        Build the request body and headers for the token request.

        :param raise_on_error: Whether to raise an error if the request cannot be built.
        :return: A tuple containing the request body and headers.
        """
        offline_token = mlrun.auth.utils.load_offline_token(
            raise_on_error=raise_on_error
        )
        if not offline_token:
            # Error already handled in `_load_offline_token`
            return None, None

        headers = {"Content-Type": "application/json"}
        request_body = {"refreshToken": offline_token}
        return request_body, headers, "json"

    def _parse_response(self, response_data):
        """
        Parse the response from the token endpoint.

        :param response_data: The JSON response data from the token endpoint.
        :param raise_on_error: Whether to raise an error if the response cannot be parsed.
        """
        spec = response_data.get("spec", {})
        access_token = spec.get("accessToken")

        if not access_token:
            raise mlrun.errors.MLRunRuntimeError(
                "Access token is missing in the response from the token endpoint"
            )

        self._token = access_token

        self._token_total_lifetime, self._token_expiry_time = (
            self._get_token_lifetime_and_expiry(access_token)
        )

    def _post_fetch_hook(self, raise_on_error=True):
        # if we reach this point and the token is non-empty but invalid,
        # it means the refresh threshold has been reached and the token will expire soon.
        if self._token and not self._is_token_within_refresh_threshold(
            cleanup_if_expired=True
        ):
            logger.warning(
                "Failed to fetch a new token. Using the existing token, which remains valid but is close to expiring."
            )

        # Perform a secondary validation that token fetch succeeded.
        # We enter this block if token fetch failed and did not raise an error
        if not self._token and raise_on_error:
            raise mlrun.errors.MLRunRuntimeError(
                "Failed to fetch a valid access token. Authentication procedure stopped."
            )

    @staticmethod
    def _get_token_lifetime_and_expiry(
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
            # already been verified earlier during the refresh access token call
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
