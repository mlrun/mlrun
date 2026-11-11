# Copyright 2026 Iguazio
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

import requests
import requests.adapters
import urllib3.util

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.http
from mlrun.utils import logger

import framework.utils.clients.helpers as clients_helpers
import framework.utils.clients.service_account_token as service_account_token
import services.api.utils.events.base as base_events

EVENTS_ACTIVATIONS_PATH = "/api/v1/events/activations"
EVENTS_REQUEST_TIMEOUT_SECONDS = 60
# Only failed connections are retried: a request that reached Orca may have published
# the event already
EVENTS_CONNECT_RETRIES = 3

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

DB_MIGRATION_REQUIRED = "MLRun.DB.Migration.Required"
DB_MIGRATION_STARTED = "MLRun.DB.Migration.Started"
DB_MIGRATION_COMPLETED = "MLRun.DB.Migration.Completed"
DB_MIGRATION_FAILED = "MLRun.DB.Migration.Failed"
DB_CONNECTION_FAILED = "MLRun.DB.Connection.Failed"

LOG_COLLECTOR_FAILED = "MLRun.LogCollector.Failed"

PROJECT_CREATION_SUCCEEDED = "MLRun.Project.Creation.Succeeded"
PROJECT_CREATION_FAILED = "MLRun.Project.Creation.Failed"
PROJECT_DELETION_SUCCEEDED = "MLRun.Project.Deletion.Succeeded"
PROJECT_DELETION_FAILED = "MLRun.Project.Deletion.Failed"

ERROR_DETAIL_LIMIT = 1024
TRUNCATION_SUFFIX = "...[truncated]"

DB_MIGRATION_EVENTS: dict[
    mlrun.common.schemas.MigrationEventActions,
    tuple[str, str],
] = {
    mlrun.common.schemas.MigrationEventActions.required: (
        DB_MIGRATION_REQUIRED,
        "MLRun database migration required, functionality may be impaired",
    ),
    mlrun.common.schemas.MigrationEventActions.started: (
        DB_MIGRATION_STARTED,
        "MLRun database migration started",
    ),
    mlrun.common.schemas.MigrationEventActions.completed: (
        DB_MIGRATION_COMPLETED,
        "MLRun database migration completed successfully",
    ),
    mlrun.common.schemas.MigrationEventActions.failed: (
        DB_MIGRATION_FAILED,
        "MLRun database migration failed, functionality may be impaired",
    ),
}

DB_CONNECTION_EVENTS: dict[
    mlrun.common.schemas.DBConnectionEventActions,
    tuple[str, str],
] = {
    mlrun.common.schemas.DBConnectionEventActions.failed: (
        DB_CONNECTION_FAILED,
        "MLRun cannot connect to its database",
    ),
}

# Description text is kept identical to the orca catalog entry for
# MLRun.LogCollector.Failed so the canonical event description stays in lockstep
# across producer and catalog; per-operation context is attached to ``details``.
LOG_COLLECTOR_EVENTS: dict[
    mlrun.common.schemas.LogCollectorEventActions,
    tuple[str, str],
] = {
    mlrun.common.schemas.LogCollectorEventActions.failed: (
        LOG_COLLECTOR_FAILED,
        "MLRun log collector failed to retrieve logs",
    ),
}

PROJECT_LIFECYCLE_EVENTS: dict[
    mlrun.common.schemas.ProjectLifecycleEventActions,
    tuple[str, str],
] = {
    mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded: (
        PROJECT_CREATION_SUCCEEDED,
        "Project was successfully created",
    ),
    mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed: (
        PROJECT_CREATION_FAILED,
        "Project creation failed",
    ),
    mlrun.common.schemas.ProjectLifecycleEventActions.deletion_succeeded: (
        PROJECT_DELETION_SUCCEEDED,
        "Project was successfully deleted",
    ),
    mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed: (
        PROJECT_DELETION_FAILED,
        "Project deletion failed",
    ),
}

PROJECT_SECRET_CREATED = "UserAction.Project.MLRunSecret.Created"
PROJECT_SECRET_UPDATED = "UserAction.Project.MLRunSecret.Updated"
PROJECT_SECRET_DELETED = "UserAction.Project.MLRunSecret.Deleted"

PROJECT_SECRET_EVENTS: dict[
    mlrun.common.schemas.SecretEventActions,
    tuple[str, str],
] = {
    mlrun.common.schemas.SecretEventActions.created: (
        PROJECT_SECRET_CREATED,
        "Project {} secret created",
    ),
    mlrun.common.schemas.SecretEventActions.updated: (
        PROJECT_SECRET_UPDATED,
        "Project {} secret updated",
    ),
    mlrun.common.schemas.SecretEventActions.deleted: (
        PROJECT_SECRET_DELETED,
        "Project {} secret deleted",
    ),
}

PERMISSION_DENIED = "UserAction.General.PermissionDenied"


class Client(base_events.BaseEventClient):
    """
    Events client for Iguazio v4. Publishes events to the Orca events API
    using catalog event configs.
    """

    def __init__(self, **_kwargs):
        self._session = self._new_session()
        self._service_account_token_client = service_account_token.Client()
        self._entity_name = self._resolve_entity_name()

    def emit(self, event: EventActivation | None):
        if event is None:
            return
        try:
            logger.debug(
                "Emitting event",
                config_name=event["configName"],
                entity_name=event["entityName"],
            )
            response = self._session.post(
                f"{mlrun.mlconf.iguazio_api_url.rstrip('/')}{EVENTS_ACTIVATIONS_PATH}",
                json=event,
                headers=clients_helpers.enrich_headers(
                    headers=self._service_account_token_client.auth_headers,
                ),
                timeout=EVENTS_REQUEST_TIMEOUT_SECONDS,
                verify=mlrun.mlconf.iguazio_api_ssl_verify,
                allow_redirects=False,
            )
            # Any non-error status counts as published
            mlrun.errors.raise_for_status(response)
        except Exception as exc:
            logger.warning(
                "Failed to emit event",
                config_name=event["configName"],
                exc=mlrun.errors.err_to_str(exc),
            )

    def generate_db_migration_event(
        self,
        action: mlrun.common.schemas.MigrationEventActions,
        error: BaseException | str | None = None,
        duration_seconds: float | None = None,
        scope: list[str] | None = None,
        versions: dict | None = None,
    ) -> EventActivation:
        try:
            config_name, description = DB_MIGRATION_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported DB migration action {action}"
            ) from exc

        details: dict[str, typing.Any] = {}
        if scope:
            details["scope"] = sorted(scope)
        if versions:
            for key, value in versions.items():
                if value is not None:
                    details[key] = value
        if duration_seconds is not None:
            details["duration_seconds"] = round(float(duration_seconds), 3)

        if action == mlrun.common.schemas.MigrationEventActions.failed:
            self._record_error(details, error)

        return self._build_event(
            config_name=config_name,
            entity_name=self._entity_name,
            description=description,
            details=details,
        )

    def generate_db_connection_event(
        self,
        action: mlrun.common.schemas.DBConnectionEventActions,
        error: BaseException | str | None = None,
        error_category: str | None = None,
        error_code: int | str | None = None,
        dialect: str | None = None,
    ) -> EventActivation:
        try:
            config_name, description = DB_CONNECTION_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported DB connection action {action}"
            ) from exc

        details: dict[str, typing.Any] = {}
        if error_category:
            details["error_category"] = error_category
        if error_code is not None:
            details["error_code"] = error_code
        if dialect:
            details["dialect"] = dialect

        self._record_error(details, error)

        return self._build_event(
            config_name=config_name,
            entity_name=self._entity_name,
            description=description,
            details=details,
        )

    def generate_log_collector_event(
        self,
        action: mlrun.common.schemas.LogCollectorEventActions,
        error: BaseException | str | None = None,
        run_uid: str | None = None,
        project: str | None = None,
    ) -> EventActivation:
        try:
            config_name, description = LOG_COLLECTOR_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported log collector action {action}"
            ) from exc

        details: dict[str, typing.Any] = {}
        if run_uid:
            details["run_uid"] = run_uid
        if project:
            details["project"] = project

        self._record_error(details, error)

        return self._build_event(
            config_name=config_name,
            entity_name=self._entity_name,
            description=description,
            details=details,
        )

    def generate_project_lifecycle_event(
        self,
        action: mlrun.common.schemas.ProjectLifecycleEventActions,
        project_name: str,
        actor: str | None = None,
        error: BaseException | str | None = None,
    ) -> EventActivation:
        try:
            config_name, description = PROJECT_LIFECYCLE_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported project lifecycle action {action}"
            ) from exc

        details: dict[str, typing.Any] = {"project_name": project_name}
        if actor:
            details["actor"] = actor

        if action in (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed,
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
        ):
            self._record_error(details, error)

        return self._build_event(
            config_name=config_name,
            entity_name=self._entity_name,
            description=description,
            details=details,
        )

    def generate_auth_secret_event(
        self,
        username: str,
        secret_name: str,
        action: mlrun.common.schemas.AuthSecretEventActions,
    ):
        # TODO: map v3 auth-secret events onto the v4 catalog (separate change).
        pass

    def generate_project_secret_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: list[str] | None = None,
        action: mlrun.common.schemas.SecretEventActions = mlrun.common.schemas.SecretEventActions.created,
        initiator: str | None = None,
    ) -> EventActivation:
        """
        Generate a project secret audit event.
        :param project: project name
        :param secret_name: secret name (unused in v4; kept for interface compatibility)
        :param secret_keys: secret keys (unused in v4; kept for interface compatibility)
        :param action: created, updated, or deleted
        :param initiator: username of the user who performed the action
        :return: event payload to emit
        """
        try:
            config_name, description_template = PROJECT_SECRET_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported project secret action {action}"
            ) from exc

        return self._build_event(
            config_name=config_name,
            kind="audit",
            class_="user action",
            project_name=project,
            entity_name=project,
            description=description_template.format(project),
            initiator=initiator,
        )

    def generate_permission_denied_event(
        self,
        resource: str,
        action: str,
        username: str | None = None,
    ) -> EventActivation:
        """
        Generate a permission-denied audit event.
        :param resource: the OPA resource path the action was attempted on
        :param action: the attempted action (e.g. read, create, update, delete)
        :param username: username of the user who was denied, if known
        :return: event payload to emit
        """
        return self._build_event(
            config_name=PERMISSION_DENIED,
            kind="audit",
            class_="user action",
            entity_name=resource,
            initiator=username,
            description=(
                f"Permission denied for user {username} to perform action "
                f"{action} in resource {resource}"
            ),
            details={"username": username, "action": action, "resource": resource},
        )

    @staticmethod
    def _build_event(
        *,
        config_name: str,
        entity_name: str,
        description: str,
        details: dict[str, typing.Any] | None = None,
        kind: str = "",
        class_: str | None = None,
        project_name: str = "",
        initiator: str | None = None,
    ) -> EventActivation:
        """
        Build the Orca event activation body. For catalog events, kind and class are left
        unset and severity is "unspecified", so Orca fills them in from the catalog entry.
        """
        event: EventActivation = {
            "configName": config_name,
            "source": "",
            "kind": kind,
            "severity": "unspecified",
            "projectName": project_name,
            "entityName": entity_name,
            "traceId": "",
            "details": details or {},
            "description": description,
        }
        if class_ is not None:
            event["class"] = class_
        if initiator is not None:
            event["initiator"] = initiator
        return event

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        session.cookies = mlrun.utils.http.DummyCookieJar()
        retry = urllib3.util.Retry(
            total=EVENTS_CONNECT_RETRIES,
            connect=EVENTS_CONNECT_RETRIES,
            read=0,
            status=0,
            other=0,
            backoff_factor=0.5,
            raise_on_status=False,
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _resolve_entity_name() -> str:
        """K8s deployment name: ``mlrun-api-{role}`` for api, ``mlrun-{name}`` otherwise."""
        service_name = mlrun.mlconf.services.service_name or "api"
        if service_name == "api":
            role = mlrun.mlconf.httpdb.clusterization.role or "chief"
            return f"mlrun-api-{role}"
        return f"mlrun-{service_name}"

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        if limit <= len(TRUNCATION_SUFFIX):
            return value[:limit]
        return value[: limit - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX

    @classmethod
    def _record_error(
        cls,
        details: dict[str, typing.Any],
        error: BaseException | str | None,
    ) -> None:
        """
        Record truncated error context in ``details``.

        The event ``description`` is intentionally left untouched: it must stay
        the generic, per-config catalog text (the events service enriches it
        from the catalog), so per-instance specifics belong only in ``details``.
        """
        if not error:
            return
        error_str = error if isinstance(error, str) else mlrun.errors.err_to_str(error)
        details["error"] = cls._truncate(error_str, ERROR_DETAIL_LIMIT)
        if not isinstance(error, str):
            details["error_type"] = type(error).__name__
