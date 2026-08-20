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
import datetime
import tempfile
import typing

import httpx
import humanfriendly
import iguazio
import requests
from iguazio.schemas import (
    RefreshAccessTokenOptionsV1,
    RefreshAccessTokensOptionsV1,
    RevokeOfflineTokenOptionsV1,
)

import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
from mlrun.utils import get_in

import framework.utils.clients.helpers as clients_helpers
import framework.utils.clients.service_account_token as service_account_token
import framework.utils.projects.remotes.leader as project_leader
from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient

_GROUP_TYPE_KEY = "@type"
_GROUP_TYPE_VALUE = "type.googleapis.com/usergroup.Group"

# Orca's project endpoints - Orca is the IG4/Oris API service, reached via the same iguazio_api_url
# this client already uses for auth/token operations.
PROJECTS_ENDPOINT = "v1/projects"
PROJECT_ENDPOINT_TEMPLATE = "v1/projects/{name}"
PROJECT_STATE_ENDPOINT_TEMPLATE = "v1/project-states/{name}"


class Client(BaseClient, project_leader.Member):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._service_account_token_client = service_account_token.Client()
        self._client = iguazio.Client(
            api_url=self._api_url,
            auto_login=False,
            use_token_file=False,
            verify_ssl=mlrun.mlconf.iguazio_api_ssl_verify,
            logger=clients_helpers.iguazio_sdk_logger(self._logger),
        )
        # Used by the Orca project-leader-proxy methods below (create/update/delete/get project), which go
        # through requests/BaseClient._send_request_to_api rather than the iguazio SDK client, since Orca's
        # project endpoints aren't modeled there.
        self._session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_exception=(
                mlrun.mlconf.httpdb.projects.retry_leader_request_on_exception
                == mlrun.common.schemas.HTTPSessionRetryMode.enabled.value
            ),
            verbose=True,
        )
        self._poll_interval_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.iguazio_project_states_poll_interval
        )
        self._poll_timeout_seconds = humanfriendly.parse_timespan(
            mlrun.mlconf.httpdb.projects.iguazio_project_states_poll_timeout
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
        self, token: str, request_headers: dict[str, str] | None = None
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
            return self._client.get_user(username).metadata.id

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
        token_name: str | None = None,
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
                    logger=clients_helpers.iguazio_sdk_logger(self._logger),
                )
                result = token_file_client.get_refresh_token()
                if not result or not result[0]:
                    self._logger.warning(
                        "No valid tokens found for user", user_id=user_id
                    )
                    raise mlrun.errors.MLRunNotFoundError(
                        "No valid tokens found for user"
                    )
                resolved_name, _ = result
                return resolved_name

            except ValueError as exc:
                # Token not found, empty, or failed validation
                self._logger.warning(
                    "Token not found or invalid for user",
                    token_name=token_name,
                    user_id=user_id,
                )
                raise mlrun.errors.MLRunNotFoundError(
                    f"Token '{token_name}' not found or invalid for user"
                ) from exc
            except RuntimeError as exc:
                # No valid tokens found after trying all
                self._logger.warning("No valid tokens found for user", user_id=user_id)
                raise mlrun.errors.MLRunNotFoundError(
                    "No valid tokens found for user"
                ) from exc

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        """
        Leader-proxy to Orca: forward project creation to Orca (relaying the acting user's identity so
        Orca's OPA authorizes it), then poll Orca's single-project state endpoint until the operation
        reaches a terminal state, to preserve the legacy synchronous ``wait_for_completion`` contract.

        :return: whether the operation is still running in the background (Orca is always async, so this
            is ``True`` unless the caller asked to wait and the operation already reached a terminal state).
        """
        self._logger.debug("Creating project in Orca", project=project.metadata.name)
        response = self._send_project_request(
            mlrun.common.types.HTTPMethod.POST,
            PROJECTS_ENDPOINT,
            "Failed creating project in Orca",
            auth_info,
            json=self._project_to_wire(project),
        )

        if not wait_for_completion:
            return True

        op_id = response.json()["status"]["op_id"]
        self._wait_for_op(project.metadata.name, op_id, auth_info)
        return False

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        # the CAS witness Orca requires for an update is the last op_id the caller observed; if the caller
        # didn't supply one, read the current state from Orca first (matches the HLD's "client reads the
        # project, then PUT/PATCH with prev_op_id" contract)
        current_op_id = (
            project.status.op_id
            or self.get_project(session, name, auth_info).status.op_id
        )
        self._logger.debug(
            "Updating project in Orca", name=name, current_op_id=current_op_id
        )
        body = self._project_to_wire(project)
        body["current_op_id"] = str(current_op_id) if current_op_id else None

        response = self._send_project_request(
            mlrun.common.types.HTTPMethod.PUT,
            PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            "Failed updating project in Orca",
            auth_info,
            json=body,
        )

        # the shared leader-client interface has no wait_for_completion for update, so always settle here to
        # match the legacy synchronous contract the caller expects
        op_id = response.json()["status"]["op_id"]
        self._wait_for_op(name, op_id, auth_info)

    def delete_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        self._logger.debug(
            "Deleting project in Orca", name=name, deletion_strategy=deletion_strategy
        )
        # the HLD documents the deletion-strategy gate as leader-side logic but doesn't pin how a caller
        # communicates the strategy to Orca; sending it in the body keeps this consistent with every other
        # write on this client until Orca's real contract lands
        response = self._send_project_request(
            mlrun.common.types.HTTPMethod.DELETE,
            PROJECT_ENDPOINT_TEMPLATE.format(name=name),
            "Failed deleting project in Orca",
            auth_info,
            json={"deletion_strategy": deletion_strategy.value},
        )

        if response.status_code == requests.codes.accepted:
            # 202: accepted, still converging - poll for completion
            if not wait_for_completion:
                return True
            # a delete's terminal signal is the project disappearing from project-states, not a status value
            op_id = response.json()["status"]["op_id"]
            self._wait_for_op(name, op_id, auth_info, absence_is_terminal=True)

        return False

    def get_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.Project:
        response = self._send_project_request(
            mlrun.common.types.HTTPMethod.GET,
            PROJECT_STATE_ENDPOINT_TEMPLATE.format(name=name),
            "Failed getting project state from Orca",
            auth_info,
        )
        return self._wire_to_project(response.json())

    def list_projects(
        self,
        session: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        updated_after: datetime.datetime | None = None,
    ) -> tuple[list[mlrun.common.schemas.Project], datetime.datetime | None]:
        # Orca-mode reconciliation is leader-driven push plus the MLRun follower's own one-time startup sync
        # (the dedicated /follower/projects/* surface); this role-2 proxy never runs mlrun's legacy
        # periodic/full project sync against Orca, so this must stay unreachable (project_sync feature gate
        # must stay disabled when leader=orca).
        raise NotImplementedError(
            "Periodic project sync is not supported when the leader is Orca"
        )

    def get_project_owner(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectOwner:
        raise NotImplementedError(
            "get_project_owner is not supported when the leader is Orca"
        )

    def format_as_leader_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.IguazioProject:
        raise NotImplementedError(
            "format_as_leader_project is not supported when the leader is Orca"
        )

    def _send_project_request(
        self,
        method: str,
        path: str,
        error_message: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        **kwargs,
    ) -> requests.Response:
        # Orca's Enterprise auth is JWT-bearer: relay the acting user's original request headers so Orca's
        # OPA authorizes on the user's own identity - mlrun never authorizes these calls itself. Deliberately
        # not using the full enrich_headers(path=...) form: it injects x-projects-role: mlrun for any
        # "projects" path, which would assert mlrun's own leader-role identity - the opposite of a pure
        # user-identity relay - so strip it explicitly even if somehow already present.
        headers = dict(auth_info.request_headers or {})
        headers.pop("content-length", None)
        headers.pop(mlrun.common.schemas.HeaderNames.projects_role, None)
        headers.update(kwargs.pop("headers", None) or {})
        kwargs["headers"] = clients_helpers.enrich_headers(headers=headers)
        kwargs.setdefault("timeout", 20)
        return self._send_request_to_api(method, path, error_message, **kwargs)

    def _wait_for_op(
        self,
        name: str,
        op_id: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        absence_is_terminal: bool = False,
    ) -> None:
        self._logger.debug(
            "Waiting for Orca operation to reach a terminal state",
            name=name,
            op_id=op_id,
        )
        mlrun.utils.helpers.retry_until_successful(
            self._poll_interval_seconds,
            self._poll_timeout_seconds,
            self._logger,
            False,
            self._verify_op_terminal,
            name,
            op_id,
            auth_info,
            absence_is_terminal,
        )

    def _verify_op_terminal(
        self,
        name: str,
        op_id: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        absence_is_terminal: bool,
    ):
        try:
            project = self.get_project("", name, auth_info)
        except mlrun.errors.MLRunNotFoundError:
            if absence_is_terminal:
                return
            raise
        if str(project.status.op_id) != str(op_id):
            # a newer operation superseded ours (e.g. a concurrent update) - nothing more to wait for
            return
        if (
            project.status.state
            not in mlrun.common.schemas.ProjectState.terminal_states()
        ):
            raise mlrun.errors.MLRunRuntimeError(
                f"Orca operation {op_id} for project {name} is still in progress "
                f"(state={project.status.state})"
            )

    @staticmethod
    def _project_to_wire(project: mlrun.common.schemas.Project) -> dict:
        # Orca's contract is the HLD's "common set" only - name/labels/annotations/owner/description -
        # deliberately excluding mlrun-specific spec (functions, artifacts, params, ...). include= keeps
        # the wire payload pinned to that set regardless of what ProjectSpec grows to contain.
        return project.model_dump(
            include={
                "metadata": {"name", "labels", "annotations"},
                "spec": {"owner", "description"},
            }
        )

    @staticmethod
    def _wire_to_project(body: dict) -> mlrun.common.schemas.Project:
        # Orca's wire shape is a strict subset of Project's own metadata/spec/status nesting - every field
        # Orca doesn't populate already has a default - so this can validate straight off the response body.
        return mlrun.common.schemas.Project.model_validate(body)

    def _extract_response_error(
        self, response: httpx.Response
    ) -> tuple[str | None, str | None]:
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

    def _prepare_request_kwargs(self, session: str | None, path: str, *, kwargs: dict):
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
        auth_headers: dict[str, str] | None = None,
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
            full_message = f"{failure_message}: {error_message}, ctx={ctx}"
            error_cls = mlrun.errors.STATUS_ERRORS.get(
                exc.response.status_code, exception_type
            )
            raise error_cls(full_message) from exc
        except Exception as exc:
            self._logger.warning(
                f"{failure_message} (unexpected error)",
                exc=mlrun.errors.err_to_str(exc),
            )
            raise exception_type(failure_message) from exc

    def _extract_ctx(self, response_body: dict) -> str | None:
        # Also used for the Orca project calls via _send_project_request/_send_request_to_api. Orca is a
        # separate Go service with an unverified error envelope (no live endpoints to check against yet) -
        # this may not parse Orca's real error responses; revisit once Orca's contract is confirmed.
        return response_body.get("status", {}).get("ctx")

    def _extract_error_message(self, response_body: dict) -> str | None:
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
