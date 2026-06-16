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

import unittest.mock

import iguazio
import pytest

import mlrun.common.schemas

import services.api.utils.events.iguazio_v4 as iguazio_v4_events


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(iguazio, "Client", unittest.mock.MagicMock())
    monkeypatch.setattr(
        "framework.utils.clients.service_account_token.Client",
        unittest.mock.MagicMock(),
    )
    mlrun.mlconf.services.service_name = "api"
    mlrun.mlconf.httpdb.clusterization.role = "chief"
    return iguazio_v4_events.Client()


@pytest.mark.parametrize(
    "action,expected_config_name",
    [
        (
            mlrun.common.schemas.MigrationEventActions.required,
            iguazio_v4_events.DB_MIGRATION_REQUIRED,
        ),
        (
            mlrun.common.schemas.MigrationEventActions.started,
            iguazio_v4_events.DB_MIGRATION_STARTED,
        ),
        (
            mlrun.common.schemas.MigrationEventActions.completed,
            iguazio_v4_events.DB_MIGRATION_COMPLETED,
        ),
        (
            mlrun.common.schemas.MigrationEventActions.failed,
            iguazio_v4_events.DB_MIGRATION_FAILED,
        ),
    ],
)
def test_generate_db_migration_event_basic(client, action, expected_config_name):
    event = client.generate_db_migration_event(action)
    assert event.config_name == expected_config_name
    assert event.entity_name == "mlrun-api-chief"
    # severity, class and kind are left unset so the orca catalog enriches them
    assert event.source == ""
    # description is the catalog default for non-failed actions
    if action != mlrun.common.schemas.MigrationEventActions.failed:
        assert event.description and "MLRun database migration" in event.description
    # error/duration are only populated when supplied
    assert "error" not in event.details
    assert "duration_seconds" not in event.details


@pytest.mark.parametrize(
    "service_name,role,expected_entity_name",
    [
        ("api", "chief", "mlrun-api-chief"),
        ("api", "worker", "mlrun-api-worker"),
        ("alerts", "chief", "mlrun-alerts"),
    ],
)
def test_entity_name_reflects_deployment(
    monkeypatch, service_name, role, expected_entity_name
):
    monkeypatch.setattr(iguazio, "Client", unittest.mock.MagicMock())
    monkeypatch.setattr(
        "framework.utils.clients.service_account_token.Client",
        unittest.mock.MagicMock(),
    )
    mlrun.mlconf.services.service_name = service_name
    mlrun.mlconf.httpdb.clusterization.role = role
    c = iguazio_v4_events.Client()
    event = c.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started
    )
    assert event.entity_name == expected_entity_name
    assert event.source == ""


def test_completed_event_carries_duration(client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.completed,
        duration_seconds=12.3456,
    )
    assert event.details == {"duration_seconds": 12.346}


@pytest.mark.parametrize(
    "scope,expected",
    [
        (["schema"], ["schema"]),
        (["data"], ["data"]),
        (["schema", "data"], ["data", "schema"]),
        (["data", "schema"], ["data", "schema"]),
        (None, None),
        ([], None),
    ],
)
def test_scope_in_details(client, scope, expected):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started,
        scope=scope,
    )
    if expected is None:
        assert "scope" not in event.details
    else:
        assert event.details["scope"] == expected


def test_versions_merged_into_details(client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.required,
        scope=["schema", "data"],
        versions={
            "current_schema_revision": "abc123",
            "target_schema_revision": "def456",
            "current_data_version": 9,
            "target_data_version": 10,
        },
    )
    assert event.details["current_schema_revision"] == "abc123"
    assert event.details["target_schema_revision"] == "def456"
    assert event.details["current_data_version"] == 9
    assert event.details["target_data_version"] == 10
    assert event.details["scope"] == ["data", "schema"]


def test_versions_drops_none_values(client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.required,
        versions={
            "current_schema_revision": "abc123",
            "target_schema_revision": "def456",
            "current_data_version": None,
            "target_data_version": None,
        },
    )
    assert event.details == {
        "current_schema_revision": "abc123",
        "target_schema_revision": "def456",
    }


def test_versions_empty_or_none(client):
    e1 = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started, versions=None
    )
    e2 = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started, versions={}
    )
    assert e1.details == {} == e2.details


def test_failed_event_with_exception_includes_summary_and_duration(client):
    err = RuntimeError("schema head mismatch")
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error=err,
        duration_seconds=4.0,
    )
    assert event.config_name == iguazio_v4_events.DB_MIGRATION_FAILED
    assert event.details["error"] == "schema head mismatch"
    assert event.details["error_type"] == "RuntimeError"
    assert event.details["duration_seconds"] == 4.0
    # description stays the generic catalog wording; per-instance error lives
    # in details only (the events service enriches description from the catalog)
    assert event.description == (
        "MLRun database migration failed, functionality may be impaired"
    )


def test_failed_event_with_string_error(client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error="boom",
    )
    assert event.details["error"] == "boom"
    # error_type is only set for exceptions, not raw strings
    assert "error_type" not in event.details
    # error is recorded in details, not appended to the generic description
    assert event.description == (
        "MLRun database migration failed, functionality may be impaired"
    )


def test_failed_event_truncates_long_error(client):
    long_err = "x" * 4096
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error=long_err,
    )
    assert event.details["error"].endswith("...[truncated]")
    # Truncation budget is hard — output must not exceed the documented limit
    assert len(event.details["error"]) <= iguazio_v4_events.ERROR_DETAIL_LIMIT
    # description is untouched by the error — stays the generic catalog text
    assert event.description == (
        "MLRun database migration failed, functionality may be impaired"
    )


@pytest.mark.parametrize("falsy_error", ["", None])
def test_failed_event_with_falsy_error_uses_catalog_description(client, falsy_error):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error=falsy_error,
    )
    # No error context appended; description stays as the catalog default,
    # and details contain neither error nor error_type.
    assert event.description == (
        "MLRun database migration failed, functionality may be impaired"
    )
    assert "error" not in event.details
    assert "error_type" not in event.details


def test_emit_calls_publish_event(client):
    spec = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started
    )
    client.emit(spec)
    client._client.with_headers.assert_called_once()
    client._client.publish_event.assert_called_once_with(spec)


def test_emit_swallows_exceptions(client):
    spec = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.required
    )
    client._client.publish_event.side_effect = RuntimeError("network down")
    client.emit(spec)


def test_emit_none_event_is_noop(client):
    client.emit(None)
    client._client.publish_event.assert_not_called()


def test_unsupported_action_raises(client):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        client.generate_db_migration_event("bogus")  # type: ignore[arg-type]


def test_generate_db_connection_event_basic(client):
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
    )
    assert event.config_name == iguazio_v4_events.DB_CONNECTION_FAILED
    assert event.entity_name == "mlrun-api-chief"
    # severity, class and kind are left unset so the orca catalog enriches them
    assert event.source == ""
    assert event.description == "MLRun cannot connect to its database"
    assert event.details == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        # falsy error — None
        {"error": None},
        # all metadata fields explicitly None
        {"error_category": None, "error_code": None, "dialect": None},
    ],
    ids=["none_error", "none_metadata"],
)
def test_db_connection_event_no_context_uses_catalog_description(client, kwargs):
    """Empty/None inputs → details stay empty and description is the catalog wording."""
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
        **kwargs,
    )
    assert event.description == "MLRun cannot connect to its database"
    assert event.details == {}


@pytest.mark.parametrize(
    "error,expected_error_substring,expected_error_type",
    [
        # Exception → carries error_type
        (
            TimeoutError("Lock wait timeout exceeded; try restarting transaction"),
            "Lock wait timeout",
            "TimeoutError",
        ),
        # Raw string → no error_type
        ("connection lost", "connection lost", None),
    ],
    ids=["exception", "string"],
)
def test_db_connection_event_renders_error(
    client, error, expected_error_substring, expected_error_type
):
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
        error=error,
        error_category="lock_wait_timeout",
        error_code=1205,
        dialect="mysql",
    )
    assert event.details["error_category"] == "lock_wait_timeout"
    assert event.details["error_code"] == 1205
    assert event.details["dialect"] == "mysql"
    assert expected_error_substring in event.details["error"]
    # error goes to details only; description stays the generic catalog text
    assert event.description == "MLRun cannot connect to its database"
    if expected_error_type is None:
        assert "error_type" not in event.details
    else:
        assert event.details["error_type"] == expected_error_type


def test_db_connection_event_truncates_long_error(client):
    long_err = "x" * 4096
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
        error=long_err,
    )
    assert event.details["error"].endswith("...[truncated]")
    assert len(event.details["error"]) <= iguazio_v4_events.ERROR_DETAIL_LIMIT
    # description is untouched by the error
    assert event.description == "MLRun cannot connect to its database"


def test_db_connection_unsupported_action_raises(client):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        client.generate_db_connection_event("bogus")  # type: ignore[arg-type]


def test_generate_log_collector_event_basic(client):
    event = client.generate_log_collector_event(
        mlrun.common.schemas.LogCollectorEventActions.failed,
    )
    assert event.config_name == iguazio_v4_events.LOG_COLLECTOR_FAILED
    assert event.entity_name == "mlrun-api-chief"
    # severity, class and kind are left unset so the orca catalog enriches them
    assert event.source == ""
    assert event.description == "MLRun log collector failed to retrieve logs"
    assert event.details == {}


def test_log_collector_event_renders_context(client):
    event = client.generate_log_collector_event(
        mlrun.common.schemas.LogCollectorEventActions.failed,
        error=RuntimeError("collector unreachable"),
        run_uid="run-7",
        project="proj-a",
    )
    assert event.details["run_uid"] == "run-7"
    assert event.details["project"] == "proj-a"
    assert event.details["error_type"] == "RuntimeError"
    assert "collector unreachable" in event.details["error"]
    # Per-instance error lives in details only; description stays the generic
    # catalog text (the events service enriches it from the catalog).
    assert event.description == "MLRun log collector failed to retrieve logs"


def test_log_collector_event_truncates_long_error(client):
    long_err = "x" * 4096
    event = client.generate_log_collector_event(
        mlrun.common.schemas.LogCollectorEventActions.failed,
        error=long_err,
    )
    assert event.details["error"].endswith("...[truncated]")
    assert len(event.details["error"]) <= iguazio_v4_events.ERROR_DETAIL_LIMIT
    # description is untouched by the error
    assert event.description == "MLRun log collector failed to retrieve logs"


def test_log_collector_unsupported_action_raises(client):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        client.generate_log_collector_event("bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "action,expected_config_name",
    [
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded,
            iguazio_v4_events.PROJECT_CREATION_SUCCEEDED,
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed,
            iguazio_v4_events.PROJECT_CREATION_FAILED,
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_succeeded,
            iguazio_v4_events.PROJECT_DELETION_SUCCEEDED,
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
            iguazio_v4_events.PROJECT_DELETION_FAILED,
        ),
    ],
)
def test_generate_project_lifecycle_event_basic(client, action, expected_config_name):
    event = client.generate_project_lifecycle_event(
        action=action, project_name="my-project", actor="alice"
    )
    assert event.config_name == expected_config_name
    # severity, class and kind are left unset so the orca catalog enriches them
    assert event.entity_name == "mlrun-api-chief"
    assert event.source == ""
    assert event.details["project_name"] == "my-project"
    assert event.details["actor"] == "alice"
    # error is only included on failed actions, and only when supplied
    assert "error" not in event.details


def test_project_lifecycle_omits_actor_when_missing(client):
    event = client.generate_project_lifecycle_event(
        action=mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded,
        project_name="p",
        actor=None,
    )
    assert event.details == {"project_name": "p"}


@pytest.mark.parametrize(
    "failed_action",
    [
        mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed,
        mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
    ],
)
def test_project_lifecycle_failed_carries_error(client, failed_action):
    err = RuntimeError("db unavailable")
    event = client.generate_project_lifecycle_event(
        action=failed_action,
        project_name="my-project",
        actor="alice",
        error=err,
    )
    assert event.details["error"] == "db unavailable"
    assert event.details["error_type"] == "RuntimeError"
    # error in details only; description stays the generic per-action catalog text
    _, expected_description = iguazio_v4_events.PROJECT_LIFECYCLE_EVENTS[failed_action]
    assert event.description == expected_description


def test_project_lifecycle_failed_truncates_long_error(client):
    long_err = "y" * 4096
    event = client.generate_project_lifecycle_event(
        action=mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
        project_name="my-project",
        actor=None,
        error=long_err,
    )
    assert event.details["error"].endswith("...[truncated]")
    assert len(event.details["error"]) <= iguazio_v4_events.ERROR_DETAIL_LIMIT
    # description is untouched by the error
    assert event.description == "Project deletion failed"


def test_project_lifecycle_succeeded_ignores_error(client):
    # even if an `error` is passed for a succeeded action it must not leak into details
    event = client.generate_project_lifecycle_event(
        action=mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded,
        project_name="my-project",
        actor="alice",
        error=RuntimeError("should be ignored"),
    )
    assert "error" not in event.details
    assert event.description == "Project was successfully created"


def test_project_lifecycle_unsupported_action_raises(client):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        client.generate_project_lifecycle_event(
            action="bogus",  # type: ignore[arg-type]
            project_name="my-project",
            actor=None,
        )
