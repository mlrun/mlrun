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


import iguazio
import iguazio.schemas

import mlrun.common.schemas
import mlrun.errors
from mlrun.utils import logger

import framework.utils.clients.helpers as clients_helpers
import framework.utils.clients.service_account_token as service_account_token
import services.api.utils.events.base as base_events

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

EVENT_KIND = "system"
EVENT_CLASS = "DB"
EVENT_CLASS_LOG_COLLECTION = "LogCollection"
EVENT_CLASS_PROJECT = "Project"
ERROR_DETAIL_LIMIT = 1024
TRUNCATION_SUFFIX = "...[truncated]"

DB_MIGRATION_EVENTS: dict[
    mlrun.common.schemas.MigrationEventActions,
    tuple[str, iguazio.schemas.Severity, str],
] = {
    mlrun.common.schemas.MigrationEventActions.required: (
        DB_MIGRATION_REQUIRED,
        iguazio.schemas.Severity.CRITICAL,
        "MLRun database migration required, functionality may be impaired",
    ),
    mlrun.common.schemas.MigrationEventActions.started: (
        DB_MIGRATION_STARTED,
        iguazio.schemas.Severity.INFO,
        "MLRun database migration started",
    ),
    mlrun.common.schemas.MigrationEventActions.completed: (
        DB_MIGRATION_COMPLETED,
        iguazio.schemas.Severity.INFO,
        "MLRun database migration completed successfully",
    ),
    mlrun.common.schemas.MigrationEventActions.failed: (
        DB_MIGRATION_FAILED,
        iguazio.schemas.Severity.CRITICAL,
        "MLRun database migration failed, functionality may be impaired",
    ),
}

DB_CONNECTION_EVENTS: dict[
    mlrun.common.schemas.DBConnectionEventActions,
    tuple[str, iguazio.schemas.Severity, str],
] = {
    mlrun.common.schemas.DBConnectionEventActions.failed: (
        DB_CONNECTION_FAILED,
        iguazio.schemas.Severity.CRITICAL,
        "MLRun cannot connect to its database",
    ),
}

# Description text is kept identical to the orca catalog entry for
# MLRun.LogCollector.Failed so the canonical event description stays in lockstep
# across producer and catalog; per-operation context is attached to ``details``.
LOG_COLLECTOR_EVENTS: dict[
    mlrun.common.schemas.LogCollectorEventActions,
    tuple[str, iguazio.schemas.Severity, str],
] = {
    mlrun.common.schemas.LogCollectorEventActions.failed: (
        LOG_COLLECTOR_FAILED,
        iguazio.schemas.Severity.MAJOR,
        "MLRun log collector failed to retrieve logs",
    ),
}

PROJECT_LIFECYCLE_EVENTS: dict[
    mlrun.common.schemas.ProjectLifecycleEventActions,
    tuple[str, iguazio.schemas.Severity, str],
] = {
    mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded: (
        PROJECT_CREATION_SUCCEEDED,
        iguazio.schemas.Severity.INFO,
        "Project was successfully created",
    ),
    mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed: (
        PROJECT_CREATION_FAILED,
        iguazio.schemas.Severity.WARNING,
        "Project creation failed",
    ),
    mlrun.common.schemas.ProjectLifecycleEventActions.deletion_succeeded: (
        PROJECT_DELETION_SUCCEEDED,
        iguazio.schemas.Severity.INFO,
        "Project was successfully deleted",
    ),
    mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed: (
        PROJECT_DELETION_FAILED,
        iguazio.schemas.Severity.WARNING,
        "Project deletion failed",
    ),
}


class Client(base_events.BaseEventClient):
    """
    Events client for Iguazio v4. Publishes events through the Iguazio (orca)
    SDK using catalog event configs.
    """

    def __init__(self, **_kwargs):
        self._client = iguazio.Client(
            api_url=mlrun.mlconf.iguazio_api_url,
            auto_login=False,
            use_token_file=False,
            verify_ssl=mlrun.mlconf.iguazio_api_ssl_verify,
        )
        self._service_account_token_client = service_account_token.Client()
        self._entity_name = self._resolve_entity_name()

    def emit(self, event):
        if event is None:
            return
        try:
            logger.debug(
                "Emitting event",
                config_name=event.config_name,
                entity_name=event.entity_name,
            )
            with self._client.with_headers(
                clients_helpers.enrich_headers(
                    headers=self._service_account_token_client.auth_headers,
                )
            ):
                self._client.publish_event(event)
        except Exception as exc:
            logger.warning(
                "Failed to emit event",
                config_name=getattr(event, "config_name", None),
                exc=mlrun.errors.err_to_str(exc),
            )

    def generate_db_migration_event(
        self,
        action: mlrun.common.schemas.MigrationEventActions,
        error: BaseException | str | None = None,
        duration_seconds: float | None = None,
        scope: list[str] | None = None,
        versions: dict | None = None,
    ) -> iguazio.schemas.EventActivationSpec:
        try:
            config_name, severity, description = DB_MIGRATION_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported DB migration action {action}"
            ) from exc

        details: dict = {}
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

        return iguazio.schemas.EventActivationSpec(
            config_name=config_name,
            source="",
            kind=EVENT_KIND,
            severity=severity,
            class_=EVENT_CLASS,
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
    ) -> iguazio.schemas.EventActivationSpec:
        try:
            config_name, severity, description = DB_CONNECTION_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported DB connection action {action}"
            ) from exc

        details: dict = {}
        if error_category:
            details["error_category"] = error_category
        if error_code is not None:
            details["error_code"] = error_code
        if dialect:
            details["dialect"] = dialect

        self._record_error(details, error)

        return iguazio.schemas.EventActivationSpec(
            config_name=config_name,
            source="",
            kind=EVENT_KIND,
            severity=severity,
            class_=EVENT_CLASS,
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
    ) -> iguazio.schemas.EventActivationSpec:
        try:
            config_name, severity, description = LOG_COLLECTOR_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported log collector action {action}"
            ) from exc

        details: dict = {}
        if run_uid:
            details["run_uid"] = run_uid
        if project:
            details["project"] = project

        self._record_error(details, error)

        return iguazio.schemas.EventActivationSpec(
            config_name=config_name,
            source="",
            kind=EVENT_KIND,
            severity=severity,
            class_=EVENT_CLASS_LOG_COLLECTION,
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
    ) -> iguazio.schemas.EventActivationSpec:
        try:
            config_name, severity, description = PROJECT_LIFECYCLE_EVENTS[action]
        except KeyError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported project lifecycle action {action}"
            ) from exc

        details: dict = {"project_name": project_name}
        if actor:
            details["actor"] = actor

        if action in (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed,
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
        ):
            self._record_error(details, error)

        return iguazio.schemas.EventActivationSpec(
            config_name=config_name,
            source="",
            kind=EVENT_KIND,
            severity=severity,
            class_=EVENT_CLASS_PROJECT,
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
        raise NotImplementedError(
            "Auth secret events are not yet supported on Iguazio v4"
        )

    def generate_project_secret_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: list[str] | None = None,
        action: mlrun.common.schemas.SecretEventActions = mlrun.common.schemas.SecretEventActions.created,
    ):
        # TODO: map v3 project-secret events onto the v4 catalog (separate change).
        raise NotImplementedError(
            "Project secret events are not yet supported on Iguazio v4"
        )

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
        details: dict,
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
