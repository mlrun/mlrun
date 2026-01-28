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
import tempfile
import typing

import httpx
import iguazio
import sqlalchemy.orm
from iguazio.schemas import (
    RefreshAccessTokenOptionsV1,
    RefreshAccessTokensOptionsV1,
    RevokeOfflineTokenOptionsV1,
    UpdateProjectOwnerOptionsV1,
)

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
from mlrun.utils import get_in

import framework.utils.clients.helpers as clients_helpers
import framework.utils.clients.service_account_token as service_account_token
import framework.utils.projects.remotes.follower as project_follower
from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient

_GROUP_TYPE_KEY = "@type"
_GROUP_TYPE_VALUE = "type.googleapis.com/usergroup.Group"


class Client(BaseClient, project_follower.Member):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._service_account_token_client = service_account_token.Client()
        self._client = iguazio.Client(
            api_url=self._api_url,
            auto_login=False,
            use_token_file=False,
            verify_ssl=mlrun.mlconf.iguazio_api_ssl_verify,
        )

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

        # Validate the offline token by sending it to Iguazio
        def _refresh_access_token():
            options = RefreshAccessTokenOptionsV1(refresh_token=secret_token.token)
            self._client.refresh_access_token(options=options)
            self._logger.info(
                "Successfully refreshed access token via Iguazio",
                token_name=secret_token.name,
            )

        return self._try_callback_with_httpx_exceptions(
            _refresh_access_token,
            mlrun.errors.MLRunUnauthorizedError,
            f"Failed to refresh access token '{secret_token.name}' from Iguazio",
        )

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

        self._logger.debug("Refreshing access tokens", token_names=token_names)

        def _refresh_access_tokens():
            options = RefreshAccessTokensOptionsV1(refresh_tokens=token_values)
            self._client.refresh_access_tokens(options=options)

        return self._try_callback_with_httpx_exceptions(
            _refresh_access_tokens,
            mlrun.errors.MLRunUnauthorizedError,
            f"Failed to refresh tokens '{', '.join(token_names)}' from Iguazio",
        )

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

        # Use Iguazio client to revoke the token
        def _revoke_offline_token():
            options = RevokeOfflineTokenOptionsV1(token=token)
            self._client.revoke_offline_token(options=options)
            self._logger.info("Successfully revoked offline token via Iguazio")

        return self._try_callback_with_httpx_exceptions(
            _revoke_offline_token,
            mlrun.errors.MLRunUnauthorizedError,
            "Failed to revoke offline token from Iguazio",
            auth_headers=request_headers,
        )

    def get_user_id_by_username(
        self,
        username: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> str:
        """
        Translate a username to user_id by querying the Iguazio management API.

        This is used when an admin wants to perform operations on another user's tokens.
        The admin provides a username, but K8s secrets are indexed by user_id.

        :param username: The username to translate to user_id.
        :param request_headers: Request headers for authentication with the Iguazio API.
        :return: The user_id corresponding to the username.
        :raises mlrun.errors.MLRunUnauthorizedError: If the request fails or the user is not found.
        """

        def _get_user_id():
            return self._client.get_user(username).metadata

        return self._try_callback_with_httpx_exceptions(
            _get_user_id,
            mlrun.errors.MLRunUnauthorizedError,
            f"Failed to get user id of '{username}' from Iguazio",
            auth_headers=auth_info.request_headers,
        )

    def resolve_token_from_igz_yml(
        self,
        igz_yml_content: str,
        user_id: str,
        token_name: typing.Optional[str] = None,
    ) -> str:
        """
        Use the iguazio SDK to resolve/validate a token from igz.yml content.

        Creates a temporary file with the provided YAML content and uses the
        Iguazio SDK's token file resolution to find and validate the token.

        :param igz_yml_content: YAML content with tokens in igz.yml format.
        :param user_id: The user_id for error messages.
        :param token_name: Specific token to validate (strict mode), or None (auto-discovery).
        :return: The resolved token name.
        :raises MLRunNotFoundError: If no valid token found or validation fails.
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=True
        ) as temp_file:
            temp_file.write(igz_yml_content)
            temp_file.flush()

            try:
                # Create a separate client configured for token file resolution
                token_file_client = iguazio.Client(
                    api_url=self._api_url,
                    auto_login=False,
                    use_token_file=True,
                    token_file_path=temp_file.name,
                    token_name=token_name,
                    verify_ssl=mlrun.mlconf.iguazio_api_ssl_verify,
                )
                result = token_file_client.get_refresh_token()
                if not result or not result[0]:
                    raise mlrun.errors.MLRunNotFoundError(
                        f"No valid tokens found for user id '{user_id}'"
                    )
                resolved_name, _ = result
                return resolved_name

            except ValueError as exc:
                # Token not found, empty, or failed validation
                raise mlrun.errors.MLRunNotFoundError(
                    f"Token '{token_name}' not found or invalid for user id '{user_id}'"
                ) from exc
            except RuntimeError as exc:
                # No valid tokens found after trying all
                raise mlrun.errors.MLRunNotFoundError(
                    f"No valid tokens found for user id '{user_id}'"
                ) from exc

    def create_project(
        self,
        session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        self._logger.debug("Creating default project policies in Iguazio")

        def _create_default_project_policies():
            # TODO: Currently, create_default_project_policies relies on the auth info of the incoming request to
            #       determine the owner of the project. The iguazio api needs to be updated to accept an explicit owner
            #       parameter so we can use the service account token here.
            #       This isn't required now, but will be for the project sync functionality.
            self._client.create_default_project_policies(project=project.metadata.name)
            self._logger.info(
                "Successfully created default project policies in Iguazio"
            )

        self._try_callback_with_httpx_exceptions(
            _create_default_project_policies,
            mlrun.errors.MLRunInternalServerError,
            "Failed to create default project policies in Iguazio",
            auth_headers=auth_info.request_headers,
        )

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        self._logger.debug(
            "Storing project owner or creating default policies in Iguazio"
        )

        def _update_owner_or_create_policies():
            try:
                # Try to create policies first
                self._client.create_default_project_policies(
                    project=project.metadata.name
                )
                self._logger.info(
                    "Successfully created default project policies in Iguazio"
                )
            except httpx.HTTPStatusError as exc:
                # If policies already exist (409 Conflict), update owner instead
                if exc.response.status_code == httpx.codes.CONFLICT:
                    self._logger.debug(
                        "Project policies already exist, updating owner instead",
                        project=project.metadata.name,
                    )
                    self.patch_project(
                        session, name, project.dict(), auth_info=auth_info
                    )
                else:
                    # Unexpected error, re-raise
                    raise

        self._try_callback_with_httpx_exceptions(
            _update_owner_or_create_policies,
            mlrun.errors.MLRunInternalServerError,
            "Failed to store project owner or create default policies in Iguazio",
            auth_headers=auth_info.request_headers,
        )

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        self._logger.debug("Updating project owner in Iguazio")

        def _update_project_owner():
            owner = project.get("spec", {}).get("owner")
            if not owner:
                # No owner to update, nothing to do
                return

            options = UpdateProjectOwnerOptionsV1(owner=owner)
            self._client.update_project_owner(project=name, options=options)
            self._logger.info("Successfully updated project owner in Iguazio")

        self._try_callback_with_httpx_exceptions(
            _update_project_owner,
            mlrun.errors.MLRunInternalServerError,
            "Failed to update project owner in Iguazio",
            auth_headers=auth_info.request_headers,
        )

    def delete_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        self._logger.debug("Deleting project policies in Iguazio")

        def _delete_project_policies():
            self._client.delete_project_policies(project=name)
            self._logger.info("Successfully deleted project policies in Iguazio")

        self._try_callback_with_httpx_exceptions(
            _delete_project_policies,
            mlrun.errors.MLRunInternalServerError,
            "Failed to delete project policies in Iguazio",
        )

    def get_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.Project:
        raise NotImplementedError("Getting a project is not supported")

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        owner: typing.Optional[str] = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: typing.Optional[list[str]] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        # TODO: This is a placeholder implementation, as it is used for project sync. Implement this method as needed
        #       when we support the project sync functionality with Iguazio 4.
        return mlrun.common.schemas.ProjectsOutput(projects=[])

    def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        owner: typing.Optional[str] = None,
        labels: typing.Optional[list[str]] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        raise NotImplementedError("Listing project summaries is not supported")

    def get_project_summary(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectSummary:
        raise NotImplementedError("Get project summary is not supported")

    def _project_policies_exist(
        self, project: str, auth_info: mlrun.common.schemas.AuthInfo
    ) -> bool:
        try:
            with self._client.with_headers(
                clients_helpers.enrich_headers(
                    headers=self._service_account_token_client.auth_headers
                )
            ):
                return self._client.get_project_policy_assignments(project=project)
        except httpx.HTTPStatusError as exc:
            error_message, ctx = self._extract_response_error(exc.response)
            if exc.response.status_code == httpx.codes.NOT_FOUND:
                self._logger.info(
                    "Project policies do not exist in Iguazio",
                    project=project,
                    error_message=error_message,
                    ctx=ctx,
                )
                return False
        except Exception as exc:
            self._logger.warning(
                "Failed to check if project policies exist in Iguazio",
                project=project,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunInternalServerError(
                "Failed to check if project policies exist in Iguazio"
            ) from exc

        return True

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
        username, user_id, group_ids, resource_type = self._parse_auth_response_data(
            response_body
        )
        return mlrun.common.schemas.AuthInfo(
            username=username,
            user_id=user_id,
            user_group_ids=group_ids,
            kind=mlrun.common.schemas.AuthInfoKind(resource_type),
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

    def _try_callback_with_httpx_exceptions(
        self,
        callback: typing.Callable[..., typing.Any],
        exception_type: type[Exception],
        failure_message: str,
        auth_headers: typing.Optional[dict[str, str]] = None,
    ) -> typing.Any:
        try:
            headers = auth_headers or self._service_account_token_client.auth_headers
            # Inject auth headers and context id to headers for logging correlation
            with self._client.with_headers(
                clients_helpers.enrich_headers(headers=headers)
            ):
                return callback()
        except httpx.HTTPStatusError as exc:
            error_message, ctx = self._extract_response_error(exc.response)
            self._logger.warning(
                failure_message,
                status_code=exc.response.status_code,
                error_message=error_message,
                ctx=ctx,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise exception_type(
                f"{failure_message}: {error_message}, ctx={ctx}"
            ) from exc
        except Exception as exc:
            self._logger.warning(
                f"{failure_message} (unexpected error)",
                exc=mlrun.errors.err_to_str(exc),
            )
            raise exception_type(failure_message) from exc

    def _extract_ctx(self, response_body: dict) -> typing.Optional[str]:
        return response_body.get("status", {}).get("ctx")

    def _extract_error_message(self, response_body: dict) -> typing.Optional[str]:
        return response_body.get("status", {}).get("errorMessage")

    @staticmethod
    def _parse_auth_response_data(
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> tuple[str, str, list[str], str]:
        """
        Validate and parse the authentication response body to extract the username, user ID, group IDs and type of
        authentication (user or service account).
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

        resource_type = get_in(
            response_body,
            "metadata.resourceType",
            mlrun.common.schemas.AuthInfoKind.user,
        )

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

        return username, user_id, group_ids, resource_type


class AsyncClient(BaseAsyncClient, Client):
    """Asynchronous implementation of the Iguazio V4 client. Inherits logic from Client and BaseAsyncClient."""

    pass
