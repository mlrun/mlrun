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
import sys
import typing

import httpx

# iguazio package is only supported in Python >= 3.11
if sys.version_info >= (3, 11):
    import iguazio
    from iguazio.schemas import (
        RefreshAccessTokenOptionsV1,
        RefreshAccessTokensOptionsV1,
        RevokeOfflineTokenOptionsV1,
    )

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
        if sys.version_info < (3, 11):
            raise mlrun.errors.MLRunRuntimeError(
                "The 'iguazio' client is only supported in Python >= 3.11"
            )
        self._client = iguazio.Client(api_url=self._api_url, auto_login=False)

    def refresh_access_token(
        self, secret_token: mlrun.common.schemas.SecretToken
    ) -> None:
        """
        Refreshes the access token by validating the provided token via the Iguazio client.

        :param secret_token: SecretToken object containing the token name and offline token string.
        :raises mlrun.errors.MLRunInvalidArgumentError: If the secret_token is None or the offline token is empty.
        :raises mlrun.errors.MLRunUnauthorizedError: If the offline token is invalid, expired, or an error
        occurs while refreshing.
        """
        if not secret_token:
            raise mlrun.errors.MLRunInvalidArgumentError("SecretToken is None")

        if not secret_token.token:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Offline token for '{secret_token.name}' is empty"
            )

        self._logger.info(
            "Refreshing access token via Iguazio", token_name=secret_token.name
        )

        try:
            # Validate the offline token by sending it to Iguazio
            options = RefreshAccessTokenOptionsV1(refresh_token=secret_token.token)
            self._client.refresh_access_token(options=options)
            self._logger.info(
                "Successfully refreshed access token via Iguazio",
                token_name=secret_token.name,
            )
        except httpx.HTTPStatusError as exc:
            error_message, ctx = self._extract_response_error(exc.response)
            self._logger.warning(
                "Failed to refresh access token from Iguazio",
                token_name=secret_token.name,
                status_code=exc.response.status_code,
                error_message=error_message,
                ctx=ctx,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunUnauthorizedError(
                f"Failed to refresh token '{secret_token.name}' from Iguazio: {error_message}, ctx={ctx}"
            ) from exc
        except Exception as exc:
            exc_str = mlrun.errors.err_to_str(exc)
            self._logger.warning(
                "Failed to refresh access token from Iguazio (unexpected error)",
                token_name=secret_token.name,
                exc=exc_str,
            )
            raise mlrun.errors.MLRunUnauthorizedError(
                f"Failed to refresh token '{secret_token.name}' from Iguazio: {exc_str}"
            ) from exc

    def refresh_access_tokens(
        self, secret_tokens: list[mlrun.common.schemas.SecretToken]
    ) -> None:
        """
        Refresh all offline tokens using the Iguazio client to validate them.

        :param secret_tokens: List of SecretToken
        :raises mlrun.errors.MLRunInvalidArgumentError: If the list is empty or any token is empty
        :raises mlrun.errors.MLRunUnauthorizedError: If any token is invalid or expired
        """
        if not secret_tokens:
            raise mlrun.errors.MLRunInvalidArgumentError("No offline tokens provided")

        token_names = [t.name for t in secret_tokens]
        token_values = [t.token for t in secret_tokens]

        if not all(token_values):
            empty_tokens = [t.name for t in secret_tokens if not t.token]
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Offline tokens are empty for: {', '.join(empty_tokens)}"
            )

        self._logger.info(
            "Refreshing multiple access tokens via Iguazio", token_names=token_names
        )

        try:
            options = RefreshAccessTokensOptionsV1(refresh_tokens=token_values)
            # Call Iguazio batch refresh
            self._client.refresh_access_tokens(options=options)

            self._logger.info(
                "Successfully refreshed multiple access tokens via Iguazio",
                token_names=token_names,
            )

        except httpx.HTTPStatusError as exc:
            error_message, ctx = self._extract_response_error(exc.response)
            self._logger.warning(
                "Failed to refresh multiple access tokens from Iguazio",
                token_names=token_names,
                status_code=exc.response.status_code,
                error_message=error_message,
                ctx=ctx,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunUnauthorizedError(
                f"Failed to refresh tokens '{', '.join(token_names)}' from Iguazio: {error_message}, ctx={ctx}"
            ) from exc
        except Exception as exc:
            exc_str = mlrun.errors.err_to_str(exc)
            self._logger.warning(
                "Failed to refresh multiple access tokens from Iguazio (unexpected error)",
                token_names=token_names,
                exc=exc_str,
            )
            raise mlrun.errors.MLRunUnauthorizedError(
                f"Failed to refresh tokens '{', '.join(token_names)}' from Iguazio: {exc_str}"
            ) from exc

    def revoke_offline_token(
        self, token: str, request_headers: typing.Optional[dict[str, str]] = None
    ) -> None:
        """
        Revoke an offline token in Iguazio.

        This method sends a revoke request to Iguazio in order to invalidate
        the provided offline token. Once revoked, the token can no longer be
        used to obtain access tokens.

        :param token: The offline token string to revoke.
        :param request_headers: Optional request headers to use for authenticating with the Iguazio management service.
        :raises mlrun.errors.MLRunInvalidArgumentError: If the provided token is empty.
        :raises mlrun.errors.MLRunUnauthorizedError: If the revocation request fails.
        """
        if not token:
            raise mlrun.errors.MLRunInvalidArgumentError("Offline token is empty")

        self._logger.info("Revoking offline token via Iguazio")

        try:
            # Use Iguazio client to revoke the token
            options = RevokeOfflineTokenOptionsV1(token=token)
            self._client.set_override_auth_headers(request_headers)
            self._client.revoke_offline_token(options=options)
            self._logger.info("Successfully revoked offline token via Iguazio")
        except httpx.HTTPStatusError as exc:
            error_message, ctx = self._extract_response_error(exc.response)
            self._logger.warning(
                "Failed to revoke offline token from Iguazio",
                status_code=exc.response.status_code,
                error_message=error_message,
                ctx=ctx,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunUnauthorizedError(
                f"Failed to revoke offline token from Iguazio: {error_message}, ctx={ctx}"
            ) from exc
        except Exception as exc:
            self._logger.warning(
                "Failed to revoke offline token from Iguazio (unexpected error)",
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunUnauthorizedError(
                "Failed to revoke offline token from Iguazio"
            ) from exc

    def _extract_response_error(
        self, response: httpx.Response
    ) -> tuple[typing.Optional[str], typing.Optional[str]]:
        """
        Extracts 'errorMessage' and 'ctx' from an Iguazio HTTP response.

        :param response: httpx.Response object from Iguazio.
        :return: Tuple of (error_message, ctx), both can be None if not present.
        """
        error_message = ctx = None
        try:
            response_body = response.json()
            error_message = self._extract_error_message(response_body)
            ctx = self._extract_ctx(response_body)
        except Exception as exc:
            self._logger.debug(
                "Failed to parse JSON from Iguazio response",
                content=response.text,
                exc=mlrun.errors.err_to_str(exc),
            )
        return error_message, ctx

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
