# Copyright 2023 Iguazio
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

import iguazio

import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
from mlrun.utils import get_in

from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient

_GROUP_TYPE_KEY = "@type"
_GROUP_TYPE_VALUE = "type.googleapis.com/group.Group"


class Client(BaseClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client = iguazio.Client(api_url=self._api_url)

    def refresh_access_token(
        self, secret_token: mlrun.common.schemas.SecretToken
    ) -> None:
        """
        Refreshes the access token by validating the provided offline token using the Iguazio client.

        :param secret_token: SecretToken object containing the token name and the offline token string.
        :raises mlrun.errors.MLRunUnauthorizedError: If the offline token is invalid or expired.
        """
        try:
            self._client.refresh_access_token(secret_token.token)
        except Exception as exc:
            raise mlrun.errors.MLRunUnauthorizedError(
                f"Failed to refresh access token '{secret_token.name}': token is invalid or expired"
            ) from exc

    def refresh_access_tokens(
        self, secret_tokens: list[mlrun.common.schemas.SecretToken]
    ) -> None:
        """
        Refresh all offline tokens using the Iguazio client to validate them.

        :param secret_tokens: List of SecretToken objects
        :raises mlrun.errors.MLRunUnauthorizedError: If any token is invalid or expired
        """
        try:
            self._client.refresh_access_tokens(secret_tokens)
        except Exception as exc:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Failed to refresh one or more access tokens: token(s) are invalid or expired"
            ) from exc

    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Extract and return AuthInfo from a valid session verification response.
        """
        username, user_id, group_ids = self._parse_auth_response_data(response_body)
        return mlrun.common.schemas.AuthInfo(
            username=username,
            user_id=user_id,
            user_group_ids=group_ids,
        )

    @property
    def _verify_session_http_method(self) -> str:
        return mlrun.common.types.HTTPMethod.GET

    def _prepare_request_kwargs(
        self, session: typing.Optional[str], path: str, *, kwargs: dict
    ):
        """
        Prepare headers for session verification request.
        Must include either an Authorization header or an _oauth2_proxy cookie.
        """
        headers = kwargs.setdefault("headers", {})

        # Accept an Authorization header or a session cookie named "_oauth2_proxy"
        authorization = headers.get(mlrun.common.schemas.HeaderNames.authorization, "")
        cookie = headers.get(mlrun.common.schemas.HeaderNames.cookie, "")

        if (
            not authorization
            and mlrun.common.schemas.CookieNames.oauth2_proxy not in cookie
        ):
            raise mlrun.errors.MLRunUnauthorizedError(
                "Request must include either an Authorization header or _oauth2_proxy cookie"
            )

    def _extract_ctx(self, response_body: dict) -> typing.Optional[str]:
        return response_body.get("status", {}).get("ctx")

    def _extract_error_message(self, response_body: dict) -> typing.Optional[str]:
        return response_body.get("status", {}).get("errorMessage")

    @staticmethod
    def _parse_auth_response_data(
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> tuple[str, str, list[str]]:
        """
        Validate and parse the authentication response body to extract the username, user ID, and group IDs.
        """
        if not isinstance(response_body, dict):
            raise mlrun.errors.MLRunBadRequestError("Expected dict in response body")

        username = get_in(response_body, "metadata.username", "")
        if not username:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Missing or empty 'metadata.username' in authentication response"
            )

        user_id = get_in(response_body, "metadata.id", "")
        if not user_id:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Missing or empty 'metadata.id' in authentication response"
            )

        group_ids = []

        relationships = response_body.get("relationships")
        if isinstance(relationships, list):
            for relationship in relationships:
                if relationship.get(_GROUP_TYPE_KEY) == _GROUP_TYPE_VALUE:
                    group_id = get_in(relationship, "metadata.id")
                    if group_id:
                        group_ids.append(group_id)
        elif relationships is not None:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Invalid format for 'relationships' in authentication response"
            )

        return username, user_id, group_ids


class AsyncClient(BaseAsyncClient, Client):
    """Asynchronous implementation of the Iguazio V4 client. Inherits logic from Client and BaseAsyncClient."""

    pass
