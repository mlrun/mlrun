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

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import requests

import mlrun.errors
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
