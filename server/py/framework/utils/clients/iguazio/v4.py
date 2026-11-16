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
import dataclasses
import datetime
import typing
import uuid

import httpx
import humanfriendly
import iguazio
import requests
import sqlalchemy.orm
import yaml
from iguazio.schemas import UpdateProjectOwnerOptionsV1

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
import mlrun.utils
import mlrun.utils.orca_client as orca_client
import mlrun.utils.orca_projects as orca_projects
from mlrun.config import config
from mlrun.utils import get_in, logger

import framework.utils.clients.helpers as clients_helpers
import framework.utils.clients.service_account_token as service_account_token
import framework.utils.projects.remotes.follower as project_follower
import framework.utils.projects.remotes.leader as project_leader
from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient

_GROUP_TYPE_KEY = "@type"
_GROUP_TYPE_VALUE = "type.googleapis.com/usergroup.Group"

# Follower Contract HLD checklist item 10: a follower triggers this on its own boot, with its own
# SA (not a user-identity relay), to have Orca run a full reconciliation sweep scoped to that
# follower. Endpoint, action type ("follower-sync", not "sync-followers" - a naming-convention
# guess that turned out wrong) and response shape (`correlationId`, not `opId`) confirmed live
# against orca's feature/ORIS-3532-trigger branch.
FOLLOWERS_SYNC_ENDPOINT = "v1/projects/followers/sync"
FOLLOWERS_SYNC_ACTION_TYPE = "follower-sync"
FOLLOWERS_SYNC_SUBDOMAIN = "projects"


@dataclasses.dataclass(frozen=True)
class ServiceAccountIdentity:
    """Send the request as MLRun's service account."""


@dataclasses.dataclass(frozen=True)
class ForwardedIdentity:
    """
    Send the request with the caller's own request headers, as-is. Nothing is added when they
    carry no Authorization header, e.g. a request authenticated by a cookie only.
    """

    request_headers: dict[str, str] | None


@dataclasses.dataclass(frozen=True)
class NoIdentity:
    """Send the request without any identity."""


OrcaIdentity = ServiceAccountIdentity | ForwardedIdentity | NoIdentity

# Body of POST /api/v1/events/activations. Functional syntax because "class" is a
# Python keyword; "class" and "initiator" are only sent when set.
EventActivation = typing.TypedDict(
    "EventActivation",
    {
        "configName": str,
        "source": str,
        "kind": str,
        "severity": str,
        "projectName": str,
        "entityName": str,
        "traceId": str,
        "details": dict[str, typing.Any],
        "description": str,
        "class": typing.NotRequired[str],
        "initiator": typing.NotRequired[str],
    },
)


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
        # Token, user and version calls
        self._orca_session = clients_helpers.new_orca_session()
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

    def refresh_access_tokens(
        self, secret_tokens: list[mlrun.common.schemas.SecretToken]
    ) -> None:
        """
        Validate offline tokens by refreshing them against Orca, as the service account.

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

        # Orca answers 207 when only some of the tokens are invalid. MLRun accepts it without reading
        # the per-token results, so a partially invalid batch is stored as-is.
        self._send_orca_request(
            method=mlrun.common.types.HTTPMethod.POST,
            path="authentication/refresh-access-tokens",
            failure_message=f"Failed to refresh tokens '{', '.join(token_names)}' from Iguazio",
            default_error_class=mlrun.errors.MLRunUnauthorizedError,
            identity=ServiceAccountIdentity(),
            json_body={"refreshTokens": token_values},
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
        :param request_headers: The caller's request headers, relayed as the identity of the request.
        :raises mlrun.errors.MLRunInvalidArgumentError: If the provided token is empty.
        :raises mlrun.errors.MLRunUnauthorizedError: If the revocation request fails.
        """
        if not token:
            raise mlrun.errors.MLRunInvalidArgumentError("Offline token is empty")

        self._logger.info("Revoking offline token via Iguazio")
        self._send_orca_request(
            method=mlrun.common.types.HTTPMethod.POST,
            path="authentication/revoke-offline-token",
            failure_message="Failed to revoke offline token from Iguazio",
            default_error_class=mlrun.errors.MLRunUnauthorizedError,
            identity=ForwardedIdentity(request_headers),
            json_body={"token": token},
        )
        self._logger.info("Successfully revoked offline token via Iguazio")

    def publish_event(self, event: EventActivation) -> None:
        """
        Publish an event activation to Orca's events API as MLRun's service account.

        :param event: The event activation to publish.
        :raises mlrun.errors.MLRunRuntimeError: If the request fails or Orca answers with a
            non-2xx status that has no matching class in ``mlrun.errors.STATUS_ERRORS``.
        """
        self._send_orca_request(
            method=mlrun.common.types.HTTPMethod.POST,
            path="events/activations",
            failure_message=f"Failed to publish event '{event['configName']}' to Iguazio",
            default_error_class=mlrun.errors.MLRunRuntimeError,
            identity=ServiceAccountIdentity(),
            json_body=event,
        )

    def get_oris_version(self) -> str:
        """
        Fetch Oris's own release version from its system-info API.

        :return: Oris's version string (e.g. "1.2.0").
        :raises mlrun.errors.MLRunRuntimeError: If the request to Oris fails or carries no version.
        """
        failure_message = "Failed to get Oris version from its system-info API"
        response = self._send_orca_request(
            method=mlrun.common.types.HTTPMethod.GET,
            path="info/system",
            failure_message=failure_message,
            default_error_class=mlrun.errors.MLRunRuntimeError,
            identity=ServiceAccountIdentity(),
        )
        return self._read_response_field(
            response=response,
            field="metadata.version.orisVersion",
            failure_message=failure_message,
            default_error_class=mlrun.errors.MLRunRuntimeError,
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
        :param auth_info: The caller's auth info, whose request headers are relayed as the identity of the request.
        :return: The user_id corresponding to the username.
        :raises mlrun.errors.MLRunUnauthorizedError: If the request fails or the user is not found.
        """
        failure_message = f"Failed to get user id of '{username}' from Iguazio"
        response = self._send_orca_request(
            method=mlrun.common.types.HTTPMethod.GET,
            path=f"profile/users/{username}",
            failure_message=failure_message,
            default_error_class=mlrun.errors.MLRunUnauthorizedError,
            identity=ForwardedIdentity(auth_info.request_headers),
        )
        return self._read_response_field(
            response=response,
            field="metadata.id",
            failure_message=failure_message,
            default_error_class=mlrun.errors.MLRunUnauthorizedError,
        )

    def resolve_token_from_igz_yml(
        self,
        igz_yml_content: str,
        user_id: str,
        token_name: str | None = None,
    ) -> str:
        """
        Resolve a token from igz.yml content and validate it against Orca.

        With a token name, that token is selected. Without one, the token named "default" is selected,
        else the first token. Only the selected token is validated, by refreshing it.

        :param igz_yml_content: YAML content with tokens in igz.yml format.
        :param user_id: The user_id for error messages.
        :param token_name: Specific token to validate (strict mode), or None (auto-discovery).
        :return: The resolved token name.
        :raises MLRunNotFoundError: If no valid token found or validation fails.
        """
        secret_tokens = (yaml.safe_load(igz_yml_content) or {}).get(
            "secretTokens"
        ) or []
        resolved_name, token_value = self._select_igz_yml_token(
            secret_tokens=secret_tokens, token_name=token_name, user_id=user_id
        )
        try:
            # Validation is unauthenticated: the refresh token itself is the credential
            self._send_orca_request(
                method=mlrun.common.types.HTTPMethod.POST,
                path="authentication/refresh-access-token",
                failure_message="Failed to validate token",
                default_error_class=mlrun.errors.MLRunNotFoundError,
                identity=NoIdentity(),
                json_body={"refreshToken": token_value},
            )
        except mlrun.errors.MLRunBaseError as exc:
            # Any failure, including Orca being unreachable, counts as an invalid token
            self._logger.warning("No valid tokens found for user", user_id=user_id)
            raise mlrun.errors.MLRunNotFoundError(
                "No valid tokens found for user"
            ) from exc
        return resolved_name

    def create_project(
        self,
        session: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        """
        Leader-proxy to Orca: forward project creation to Orca (relaying the acting user's identity so
        Orca's OPA authorizes it), then poll the project-sync trackable action for the returned op_id
        until it reaches a terminal state, to preserve the legacy synchronous ``wait_for_completion``
        contract.

        :return: whether the operation is still running in the background (Orca is always async, so this
            is ``True`` unless the caller asked to wait and the operation already reached a terminal state).
        """
        orchestrator = self._orca_orchestrator(auth_info)
        _, op_id = orchestrator.create(project)
        return self._settle_create(
            orchestrator=orchestrator,
            name=project.metadata.name,
            op_id=op_id,
            wait_for_completion=wait_for_completion,
        )

    def store_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        wait_for_completion: bool = True,
    ) -> bool:
        """
        Leader-proxy to Orca: create or update the project depending on Orca's state of it (see
        :meth:`mlrun.utils.orca_client.ProjectsOrchestrator.store`), not MLRun's local DB - a project
        MLRun has already prepared locally may still be ``creating`` in Orca, and only a create resumes it.

        :return: whether the operation is still running in the background.
        """
        orchestrator = self._orca_orchestrator(auth_info)
        response, op_id, created = orchestrator.store(name, project)
        if created:
            return self._settle_create(
                orchestrator=orchestrator,
                name=name,
                op_id=op_id,
                wait_for_completion=wait_for_completion,
            )
        self._settle_update(
            orchestrator=orchestrator, name=name, response=response, op_id=op_id
        )
        return False

    def update_project(
        self,
        session: str,
        name: str,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        orchestrator = self._orca_orchestrator(auth_info)
        response, op_id = orchestrator.update(name, project)
        self._settle_update(
            orchestrator=orchestrator, name=name, response=response, op_id=op_id
        )

    def delete_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
        wait_for_completion: bool = True,
    ) -> bool:
        if deletion_strategy == mlrun.common.schemas.DeletionStrategy.check:
            # Orca's DeleteProjectOptions has no strategy concept yet - a plain DELETE always deletes,
            # so a "check-only" caller must not fall through to that below.
            return False
        orchestrator = self._orca_orchestrator(auth_info)
        response = orchestrator.delete(name)

        if response.status_code == requests.codes.accepted:
            # 202: accepted, still converging - poll for completion
            if not wait_for_completion:
                return True
            op_id = response.json()["status"]["opId"]
            orchestrator.wait_for_op(name, op_id)

        return False

    def get_project(
        self,
        session: str,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.Project:
        return self._orca_orchestrator(auth_info).get(name)

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

    def _settle_create(
        self,
        orchestrator: orca_client.ProjectsOrchestrator,
        name: str,
        op_id: uuid.UUID | str,
        wait_for_completion: bool,
    ) -> bool:
        if not wait_for_completion:
            return True
        # this leader-proxy only ever needs to know the operation settled, not the resulting
        # project (unlike the SDK-direct client, which returns it) - wait_for_op alone, not
        # settle(), which would also fetch it via an extra, unneeded GET.
        try:
            orchestrator.wait_for_op(name, op_id)
        except mlrun.errors.MLRunRetryExhaustedError as exc:
            # A follower failing its step doesn't roll the create back - Orca keeps the project
            # creating and its reconciler resumes the operation, so it is still in progress.
            if (
                orca_client.is_action_failed(exc)
                and orchestrator.get(name).status.state
                == mlrun.common.schemas.ProjectState.creating
            ):
                self._logger.warning(
                    "Orca project create action failed, leaving it to Orca's reconciliation",
                    name=name,
                    op_id=op_id,
                    exc=mlrun.errors.err_to_str(exc.__cause__),
                )
                return True
            raise
        return False

    @staticmethod
    def _settle_update(
        orchestrator: orca_client.ProjectsOrchestrator,
        name: str,
        response: requests.Response,
        op_id: uuid.UUID | str,
    ) -> None:
        # the shared leader-client interface has no wait_for_completion for update, so always wait here
        # to match the legacy synchronous contract the caller expects. A synchronous (non-202) response
        # is already terminal - nothing to poll for. A follower missing an update Orca already stored
        # isn't a failure, but Orca may turn a PUT into a create, which must stay strict.
        if response.status_code == requests.codes.accepted:
            orchestrator.wait_for_op(
                name,
                op_id,
                tolerate_failed_fan_out=orca_projects.is_update_of_existing_project(
                    response.json()
                ),
            )

    def _orca_orchestrator(
        self, auth_info: mlrun.common.schemas.AuthInfo
    ) -> orca_client.ProjectsOrchestrator:
        # auth_info changes per call (this relays the acting user's identity), so a fresh
        # orchestrator is built per call rather than cached on self like _session/
        # _poll_interval_seconds - it's a cheap object holding no I/O state of its own.
        return orca_client.ProjectsOrchestrator(
            lambda method, path, error_message, **kwargs: self._send_project_request(
                method, path, error_message, auth_info, **kwargs
            ),
            self._logger,
            self._poll_interval_seconds,
            self._poll_timeout_seconds,
        )

    def trigger_followers_sync(self) -> None:
        """
        Triggers Orca's leader-side reconciliation sweep scoped to this follower (MLRun's own
        SA - not a user-identity relay, since this runs on boot, outside any request) and blocks
        until the resulting Trackable Action execution reaches a terminal state. Per the Follower
        Contract HLD (checklist item 10), MLRun never reads a states list or computes its own
        diff here - Orca pushes truth back onto MLRun's own ``/follower/projects/*`` surface
        while this sweep runs.
        """
        self._logger.debug("Triggering Orca followers-sync")
        response = self._send_self_authenticated_request(
            mlrun.common.types.HTTPMethod.POST,
            FOLLOWERS_SYNC_ENDPOINT,
            "Failed triggering Orca followers-sync",
        )
        # Response shape confirmed live against orca's feature/ORIS-3532-trigger branch:
        # {"correlationId": "<uuid>", "follower": "mlrun", "status": {...}} - note this is
        # `correlationId`, not `opId`; unlike a project's own op_id, this identifies the sweep
        # run itself, which is what the trackable-action poll below keys on.
        correlation_id = response.json().get("correlationId")
        if not correlation_id:
            raise mlrun.errors.MLRunRuntimeError(
                "Orca followers-sync response did not include a correlationId"
            )
        self._wait_for_followers_sync(correlation_id)

    def _send_self_authenticated_request(
        self,
        method: str,
        path: str,
        error_message: str,
        **kwargs,
    ) -> requests.Response:
        # Unlike _send_project_request (which relays the acting user's identity so Orca's OPA
        # authorizes on the user), this call has no user in the loop at all - it runs on boot,
        # authenticated as MLRun's own SA.
        headers = self._service_account_token_client.escalate_request_headers(
            kwargs.pop("headers", None)
        )
        # escalate_request_headers sets the HTTP-conventional "Authorization" casing; mirror it
        # under the lowercase HeaderNames.authorization key too, since _prepare_request_kwargs's
        # pre-flight check looks up that exact (lowercase) key.
        headers.setdefault(
            mlrun.common.schemas.HeaderNames.authorization,
            headers.get("Authorization"),
        )
        kwargs["headers"] = clients_helpers.enrich_headers(headers=headers)
        kwargs.setdefault("timeout", 20)
        return self._send_request_to_api(method, path, error_message, **kwargs)

    def _wait_for_followers_sync(self, correlation_id: str) -> None:
        self._logger.debug(
            "Waiting for Orca followers-sync action to reach a terminal state",
            correlation_id=correlation_id,
        )
        try:
            mlrun.utils.helpers.retry_until_successful(
                self._poll_interval_seconds,
                self._poll_timeout_seconds,
                self._logger,
                False,
                self._verify_followers_sync_terminal,
                correlation_id,
                fatal_exceptions=(orca_projects.OrcaActionFailedError,),
            )
        except orca_projects.OrcaActionFailedError as exc:
            raise mlrun.errors.MLRunRuntimeError(str(exc)) from exc

    def _verify_followers_sync_terminal(self, correlation_id: str) -> None:
        response = self._send_self_authenticated_request(
            mlrun.common.types.HTTPMethod.GET,
            orca_projects.ACTION_EXECUTIONS_ENDPOINT,
            "Failed getting Orca followers-sync action execution",
            params={
                "correlationId": str(correlation_id),
                "actionType": FOLLOWERS_SYNC_ACTION_TYPE,
                "subdomain": FOLLOWERS_SYNC_SUBDOMAIN,
                "limit": 1,
            },
        )
        items = response.json().get("items", [])
        if not items:
            raise mlrun.errors.MLRunRuntimeError(
                f"No Orca followers-sync action observed yet (correlation_id={correlation_id})"
            )
        state = items[0].get("status", {}).get("state")
        if state == "failed":
            raise orca_projects.OrcaActionFailedError(
                f"Orca followers-sync action (correlation_id={correlation_id}) failed"
            )
        if state != "succeeded":
            raise mlrun.errors.MLRunRuntimeError(
                f"Orca followers-sync action (correlation_id={correlation_id}) is still in "
                f"progress (state={state})"
            )

    def _extract_response_error(
        self, response: httpx.Response | requests.Response
    ) -> tuple[str | None, str | None]:
        """
        Extracts 'errorMessage' and 'ctx' from an Iguazio HTTP response.

        :param response: Response from Iguazio.
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

    def _send_orca_request(
        self,
        method: str,
        path: str,
        failure_message: str,
        default_error_class: type[Exception],
        identity: OrcaIdentity,
        json_body: dict | None = None,
    ) -> requests.Response:
        """
        Send a request to Orca's /api/v1 API with an explicit identity.

        :param method: HTTP method.
        :param path: Path under /api/v1.
        :param failure_message: Message of the raised error.
        :param default_error_class: Raised on a transport failure, or on an error status with no
            matching class in ``mlrun.errors.STATUS_ERRORS``.
        :param identity: Who the request is sent as.
        :param json_body: JSON body of the request.
        :return: The response, for any 2xx status.
        """
        try:
            match identity:
                case ServiceAccountIdentity():
                    headers = self._service_account_token_client.auth_headers
                case ForwardedIdentity(request_headers=request_headers):
                    headers = dict(request_headers or {})
                    headers.pop("content-length", None)
                case NoIdentity():
                    headers = {}
                case _:
                    raise TypeError(
                        f"Unsupported Orca identity: {type(identity).__name__}"
                    )
            response = self._orca_session.request(
                method,
                f"{self._api_url}/api/v1/{path}",
                headers=clients_helpers.enrich_headers(headers=headers),
                json=json_body,
                timeout=clients_helpers.ORCA_REQUEST_TIMEOUT_SECONDS,
                verify=mlrun.mlconf.iguazio_api_ssl_verify,
                allow_redirects=False,
            )
        except (
            requests.RequestException,
            # the service-account token can't be read, or the cached one is malformed
            OSError,
            mlrun.errors.MLRunInvalidArgumentError,
        ) as exc:
            self._logger.warning(
                "Request to Orca failed",
                failure_message=failure_message,
                path=path,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise default_error_class(failure_message) from exc

        # Redirects aren't followed, so a 3xx fails too
        if not 200 <= response.status_code < 300:
            error_message, ctx = self._extract_response_error(response)
            self._logger.warning(
                "Request to Orca failed",
                failure_message=failure_message,
                path=path,
                status_code=response.status_code,
                error_message=error_message,
                ctx=ctx,
            )
            error_class = mlrun.errors.STATUS_ERRORS.get(
                response.status_code, default_error_class
            )
            raise error_class(f"{failure_message}: {error_message}, ctx={ctx}")
        return response

    def _read_response_field(
        self,
        response: requests.Response,
        field: str,
        failure_message: str,
        default_error_class: type[Exception],
    ) -> str:
        try:
            value = get_in(response.json(), field)
        except ValueError:
            value = None
        if not value or not isinstance(value, str):
            self._logger.warning(
                "Orca response is missing a field",
                field=field,
                path=response.request.path_url,
            )
            raise default_error_class(failure_message)
        return value

    def _select_igz_yml_token(
        self,
        secret_tokens: list[dict],
        token_name: str | None,
        user_id: str,
    ) -> tuple[str, str]:
        if token_name:
            entry = next(
                (token for token in secret_tokens if token.get("name") == token_name),
                None,
            )
            if not entry or not entry.get("token"):
                self._logger.warning(
                    "Token not found or invalid for user",
                    token_name=token_name,
                    user_id=user_id,
                )
                raise mlrun.errors.MLRunNotFoundError(
                    f"Token '{token_name}' not found or invalid for user"
                )
            return token_name, entry["token"]

        default_entries = [
            token for token in secret_tokens if token.get("name") == "default"
        ]
        # Several "default" entries are ambiguous, so none of them is selected
        entry = None
        if len(default_entries) == 1:
            entry = default_entries[0]
        elif not default_entries and secret_tokens:
            entry = secret_tokens[0]
        if not entry or not entry.get("name") or not entry.get("token"):
            self._logger.warning("No valid tokens found for user", user_id=user_id)
            raise mlrun.errors.MLRunNotFoundError("No valid tokens found for user")
        return entry["name"], entry["token"]

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
        # Also used for the Orca project calls via _send_project_request/_send_request_to_api.
        return orca_projects.extract_error_details(response_body)[1]

    def _extract_error_message(self, response_body: dict) -> str | None:
        return orca_projects.extract_error_details(response_body)[0]

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


class FollowerClient(Client, project_follower.Member):
    """
    Iguazio as a project follower, used when MLRun is the projects leader (the "igz" follower):
    keeps the project's default policies in Iguazio in sync with MLRun's projects.

    Subclasses the Orca-leader ``Client`` only to reuse its SDK client and error handling - it is not
    a usable leader member.
    """

    def create_project(
        self,
        session: sqlalchemy.orm.Session,
        project: mlrun.common.schemas.Project,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ):
        self._logger.debug("Creating default project policies in Iguazio")

        def _create_default_project_policies():
            # Iguazio takes the project owner from the caller's identity, so this relays the
            # incoming request's auth headers rather than using the service account token.
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
            "Ensuring default project policies exist in Iguazio", project=name
        )

        def _create_policies_if_not_exist():
            try:
                self._client.create_default_project_policies(project=name)
                self._logger.info(
                    "Successfully created default project policies in Iguazio",
                    project=name,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == httpx.codes.CONFLICT:
                    # Conflict means policies already exist, which is expected when storing
                    # an existing project (e.g., load_project, sync, or re-save).
                    # We use "create and handle conflict" instead of "check then create"
                    # because checking a non-existent project returns 401, not 404.
                    self._logger.debug(
                        "Project policies already exist, skipping", project=name
                    )
                else:
                    # Unexpected error, re-raise
                    raise

        self._try_callback_with_httpx_exceptions(
            _create_policies_if_not_exist,
            mlrun.errors.MLRunInternalServerError,
            "Failed to store project policies in Iguazio",
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
        if deletion_strategy == mlrun.common.schemas.DeletionStrategy.check:
            return

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
        owner: str | None = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
        updated_after: datetime.datetime | None = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        # Iguazio holds only project policies, no project records, so periodic project sync would see every
        # project as missing here and retry creating its policies - deployments keep that sync disabled
        # (periodic_sync_interval=0s).
        return mlrun.common.schemas.ProjectsOutput(projects=[])

    def list_project_summaries(
        self,
        session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
        owner: str | None = None,
        labels: list[str] | None = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] | None = None,
    ) -> mlrun.common.schemas.ProjectSummariesOutput:
        raise NotImplementedError("Listing project summaries is not supported")

    def get_project_summary(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        auth_info: mlrun.common.schemas.AuthInfo = mlrun.common.schemas.AuthInfo(),
    ) -> mlrun.common.schemas.ProjectSummary:
        raise NotImplementedError("Get project summary is not supported")


class AsyncClient(BaseAsyncClient, Client):
    """Asynchronous implementation of the Iguazio V4 client. Inherits logic from Client and BaseAsyncClient."""

    pass


# if oris version specified on mlrun config set it likewise,
# if not specified, get it from Oris's own info API
# since this is a heavy operation (sending requests to API), and it's unlikely that the version
# will change - only fetch it once
def resolve_oris_version() -> str:
    if not config.oris_version and config.iguazio_api_url:
        try:
            config.oris_version = Client().get_oris_version()
        except Exception as exc:
            logger.warning(
                "Failed to resolve Oris version", exc=mlrun.errors.err_to_str(exc)
            )

    return config.oris_version
