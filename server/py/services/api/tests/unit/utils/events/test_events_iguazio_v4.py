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
import iguazio.schemas
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
    "action,expected_config_name,expected_severity",
    [
        (
            mlrun.common.schemas.MigrationEventActions.required,
            iguazio_v4_events.DB_MIGRATION_REQUIRED,
            iguazio.schemas.Severity.CRITICAL,
        ),
        (
            mlrun.common.schemas.MigrationEventActions.started,
            iguazio_v4_events.DB_MIGRATION_STARTED,
            iguazio.schemas.Severity.INFO,
        ),
        (
            mlrun.common.schemas.MigrationEventActions.completed,
            iguazio_v4_events.DB_MIGRATION_COMPLETED,
            iguazio.schemas.Severity.INFO,
        ),
        (
            mlrun.common.schemas.MigrationEventActions.failed,
            iguazio_v4_events.DB_MIGRATION_FAILED,
            iguazio.schemas.Severity.CRITICAL,
        ),
    ],
)
def test_generate_db_migration_event_basic(
    client, action, expected_config_name, expected_severity
):
    event = client.generate_db_migration_event(action)
    assert event.config_name == expected_config_name
    assert event.kind == "system"
    assert event.class_ == "Platform"
    assert event.severity == expected_severity
    assert event.entity_name == "MLRun"
    assert event.source == "mlrun-api-chief"
    # description is the catalog default for non-failed actions
    if action != mlrun.common.schemas.MigrationEventActions.failed:
        assert event.description and "MLRun database migration" in event.description
    # error/duration are only populated when supplied
    assert "error" not in event.details
    assert "duration_seconds" not in event.details


@pytest.mark.parametrize(
    "service_name,role,expected_source",
    [
        ("api", "chief", "mlrun-api-chief"),
        ("api", "worker", "mlrun-api-worker"),
        ("alerts", "chief", "mlrun-alerts"),
    ],
)
def test_source_reflects_deployment(monkeypatch, service_name, role, expected_source):
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
    assert event.source == expected_source


def test_completed_event_carries_duration(client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.completed,
        duration_seconds=12.3456,
    )
    assert event.details == {"duration_seconds": 12.346}
    assert event.severity == iguazio.schemas.Severity.INFO


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
    assert "schema head mismatch" in event.description
    # description still contains the catalog wording
    assert "MLRun database migration failed" in event.description


def test_failed_event_with_string_error(client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error="boom",
    )
    assert event.details["error"] == "boom"
    # error_type is only set for exceptions, not raw strings
    assert "error_type" not in event.details
    assert "boom" in event.description


def test_failed_event_truncates_long_error(client):
    long_err = "x" * 4096
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error=long_err,
    )
    assert event.details["error"].endswith("...[truncated]")
    # Truncation budget is hard — output must not exceed the documented limit
    assert len(event.details["error"]) <= iguazio_v4_events.ERROR_DETAIL_LIMIT
    # description summary is truncated more aggressively
    assert event.description.endswith("...[truncated]")
    assert len(event.description) < 350


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
