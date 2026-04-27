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

DB_MIGRATION_REQUIRED = "Platform.MLRun.DB.MigrationRequired"
DB_MIGRATION_STARTED = "Platform.MLRun.DB.MigrationStarted"
DB_MIGRATION_COMPLETED = "Platform.MLRun.DB.MigrationCompleted"
DB_MIGRATION_FAILED = "Platform.MLRun.DB.MigrationFailed"

ENTITY_NAME = "MLRun"
EVENT_KIND = "system"
EVENT_CLASS = "Platform"
ERROR_DETAIL_LIMIT = 1024
ERROR_DESCRIPTION_LIMIT = 200
TRUNCATION_SUFFIX = "...[truncated]"

# Per IG4 System Events Spec — Phase 2.
# class is "Platform", kind is "system" for all DB lifecycle events.
# Severity follows the spec: Required/Failed are Critical, Started/Completed are Info.
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


class Client(base_events.BaseEventClient):
    """
    Events client for Iguazio v4 — publishes events through the Iguazio (orca) SDK
    using catalog event configs.
    """

    def __init__(self, **_kwargs):
        self._client = iguazio.Client(
            api_url=mlrun.mlconf.iguazio_api_url,
            auto_login=False,
            use_token_file=False,
            verify_ssl=mlrun.mlconf.iguazio_api_ssl_verify,
        )
        self._service_account_token_client = service_account_token.Client()
        self._source = self._resolve_source()

    def emit(self, event):
        if event is None:
            return
        try:
            logger.debug(
                "Emitting event",
                config_name=event.config_name,
                source=event.source,
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

        if action == mlrun.common.schemas.MigrationEventActions.failed and error:
            error_str = (
                error if isinstance(error, str) else mlrun.errors.err_to_str(error)
            )
            details["error"] = self._truncate(error_str, ERROR_DETAIL_LIMIT)
            if not isinstance(error, str):
                details["error_type"] = type(error).__name__
            description = (
                f"{description}: {self._truncate(error_str, ERROR_DESCRIPTION_LIMIT)}"
            )

        return iguazio.schemas.EventActivationSpec(
            config_name=config_name,
            source=self._source,
            kind=EVENT_KIND,
            severity=severity,
            class_=EVENT_CLASS,
            entity_name=ENTITY_NAME,
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
    def _resolve_source() -> str:
        """
        Use the K8s deployment name as the event source so consumers can tell which
        component emitted the event:
        * the api service runs as `mlrun-api-chief` or `mlrun-api-worker`
          (role is set in mlconf.httpdb.clusterization.role)
        * other services (currently just alerts) deploy as `mlrun-<service_name>`
        """
        service_name = mlrun.mlconf.services.service_name or "api"
        if service_name == "api":
            role = mlrun.mlconf.httpdb.clusterization.role or "chief"
            return f"mlrun-api-{role}"
        return f"mlrun-{service_name}"

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        """Trim ``value`` so the returned string is at most ``limit`` characters."""
        if len(value) <= limit:
            return value
        if limit <= len(TRUNCATION_SUFFIX):
            return value[:limit]
        return value[: limit - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
