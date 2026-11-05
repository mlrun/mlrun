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
"""
Wire-level contract of the events MLRun publishes to Orca.

A local HTTP server stands in for Orca. Every test generates an event, emits it, and asserts
on the JSON body Orca receives, so the tests don't depend on what the generators return.
"""

import socket
import unittest.mock

import pytest
import pytest_httpserver
import requests
import urllib3.connection
import urllib3.util

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
from mlrun.utils.logger import context_id_var

import framework.utils.clients.service_account_token
import services.api.utils.events.iguazio_v4 as iguazio_v4_events

SA_TOKEN = "sa-token"
CTX_ID = "test-context-id"
EVENTS_PATH = "/api/v1/events/activations"
LONG_ERROR = "x" * 4096
TRUNCATED_ERROR = "x" * 1010 + "...[truncated]"

MIGRATION_FAILED_DESCRIPTION = (
    "MLRun database migration failed, functionality may be impaired"
)
DB_CONNECTION_DESCRIPTION = "MLRun cannot connect to its database"
LOG_COLLECTOR_DESCRIPTION = "MLRun log collector failed to retrieve logs"
# What Orca returns for a published event
ACTIVATION_RESPONSE = {
    "metadata": {"id": "activation-1"},
    "spec": {"configName": "MLRun.DB.Migration.Started"},
    "status": {},
}


@pytest.fixture(autouse=True)
def service_account_token(monkeypatch):
    monkeypatch.setattr(
        framework.utils.clients.service_account_token.Client,
        "token",
        property(lambda self: SA_TOKEN),
    )


@pytest.fixture
def orca(httpserver: pytest_httpserver.HTTPServer):
    mlrun.mlconf.iguazio_api_url = httpserver.url_for("").rstrip("/")
    yield httpserver
    httpserver.check_assertions()


@pytest.fixture
def unreachable_orca():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mlrun.mlconf.iguazio_api_url = f"http://127.0.0.1:{port}"


@pytest.fixture
def context_id():
    reset_token = context_id_var.set(CTX_ID)
    yield CTX_ID
    context_id_var.reset(reset_token)


@pytest.fixture
def warning_log():
    with unittest.mock.patch.object(mlrun.utils.logger, "warning") as warning:
        yield warning


@pytest.fixture
def client(orca):
    return _make_client()


# ---------------------------------------------------------------------------
# DB migration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,config_name,description",
    [
        (
            mlrun.common.schemas.MigrationEventActions.required,
            "MLRun.DB.Migration.Required",
            "MLRun database migration required, functionality may be impaired",
        ),
        (
            mlrun.common.schemas.MigrationEventActions.started,
            "MLRun.DB.Migration.Started",
            "MLRun database migration started",
        ),
        (
            mlrun.common.schemas.MigrationEventActions.completed,
            "MLRun.DB.Migration.Completed",
            "MLRun database migration completed successfully",
        ),
        (
            mlrun.common.schemas.MigrationEventActions.failed,
            "MLRun.DB.Migration.Failed",
            MIGRATION_FAILED_DESCRIPTION,
        ),
    ],
)
def test_db_migration_event_payload(orca, client, action, config_name, description):
    body = _published_body(orca, client, client.generate_db_migration_event(action))
    assert body == _system_event(config_name, description, details={})


@pytest.mark.parametrize(
    "service_name,role,expected_entity_name",
    [
        ("api", "chief", "mlrun-api-chief"),
        ("api", "worker", "mlrun-api-worker"),
        ("alerts", "chief", "mlrun-alerts"),
    ],
)
def test_entity_name_reflects_deployment(
    orca, service_name, role, expected_entity_name
):
    client = _make_client(service_name=service_name, role=role)
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started
    )
    body = _published_body(orca, client, event)
    assert body == _system_event(
        "MLRun.DB.Migration.Started",
        "MLRun database migration started",
        details={},
        entity_name=expected_entity_name,
    )


def test_db_migration_completed_carries_duration(orca, client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.completed,
        duration_seconds=12.3456,
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {"duration_seconds": 12.346}


@pytest.mark.parametrize(
    "scope,expected_details",
    [
        (["schema"], {"scope": ["schema"]}),
        (["data"], {"scope": ["data"]}),
        (["schema", "data"], {"scope": ["data", "schema"]}),
        (["data", "schema"], {"scope": ["data", "schema"]}),
        (None, {}),
        ([], {}),
    ],
)
def test_db_migration_scope_in_details(orca, client, scope, expected_details):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started,
        scope=scope,
    )
    body = _published_body(orca, client, event)
    assert body["details"] == expected_details


def test_db_migration_versions_merged_into_details(orca, client):
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
    body = _published_body(orca, client, event)
    assert body["details"] == {
        "scope": ["data", "schema"],
        "current_schema_revision": "abc123",
        "target_schema_revision": "def456",
        "current_data_version": 9,
        "target_data_version": 10,
    }


def test_db_migration_versions_drop_none_values(orca, client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.required,
        versions={
            "current_schema_revision": "abc123",
            "target_schema_revision": "def456",
            "current_data_version": None,
            "target_data_version": None,
        },
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {
        "current_schema_revision": "abc123",
        "target_schema_revision": "def456",
    }


@pytest.mark.parametrize("versions", [None, {}])
def test_db_migration_versions_empty_or_none(orca, client, versions):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.started,
        versions=versions,
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {}


def test_db_migration_failed_with_exception(orca, client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error=RuntimeError("schema head mismatch"),
        duration_seconds=4.0,
    )
    body = _published_body(orca, client, event)
    # The per-instance error goes to details only; the description stays the catalog text.
    assert body == _system_event(
        "MLRun.DB.Migration.Failed",
        MIGRATION_FAILED_DESCRIPTION,
        details={
            "duration_seconds": 4.0,
            "error": "schema head mismatch",
            "error_type": "RuntimeError",
        },
    )


def test_db_migration_failed_with_string_error(orca, client):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error="boom",
    )
    body = _published_body(orca, client, event)
    # error_type is only set for exceptions, not raw strings
    assert body == _system_event(
        "MLRun.DB.Migration.Failed",
        MIGRATION_FAILED_DESCRIPTION,
        details={"error": "boom"},
    )


@pytest.mark.parametrize("falsy_error", ["", None])
def test_db_migration_failed_falsy_error(orca, client, falsy_error):
    event = client.generate_db_migration_event(
        mlrun.common.schemas.MigrationEventActions.failed,
        error=falsy_error,
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {}
    assert body["description"] == MIGRATION_FAILED_DESCRIPTION


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------


def test_db_connection_event_payload(orca, client):
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
    )
    body = _published_body(orca, client, event)
    assert body == _system_event(
        "MLRun.DB.Connection.Failed", DB_CONNECTION_DESCRIPTION, details={}
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": None},
        {"error_category": None, "error_code": None, "dialect": None},
    ],
    ids=["none_error", "none_metadata"],
)
def test_db_connection_no_context(orca, client, kwargs):
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
        **kwargs,
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {}
    assert body["description"] == DB_CONNECTION_DESCRIPTION


@pytest.mark.parametrize(
    "error,expected_error_fields",
    [
        (
            TimeoutError("Lock wait timeout exceeded; try restarting transaction"),
            {
                "error": "Lock wait timeout exceeded; try restarting transaction",
                "error_type": "TimeoutError",
            },
        ),
        ("connection lost", {"error": "connection lost"}),
    ],
    ids=["exception", "string"],
)
def test_db_connection_renders_error(orca, client, error, expected_error_fields):
    event = client.generate_db_connection_event(
        mlrun.common.schemas.DBConnectionEventActions.failed,
        error=error,
        error_category="lock_wait_timeout",
        error_code=1205,
        dialect="mysql",
    )
    body = _published_body(orca, client, event)
    assert body == _system_event(
        "MLRun.DB.Connection.Failed",
        DB_CONNECTION_DESCRIPTION,
        details={
            "error_category": "lock_wait_timeout",
            "error_code": 1205,
            "dialect": "mysql",
            **expected_error_fields,
        },
    )


# ---------------------------------------------------------------------------
# Log collector
# ---------------------------------------------------------------------------


def test_log_collector_event_payload(orca, client):
    event = client.generate_log_collector_event(
        mlrun.common.schemas.LogCollectorEventActions.failed,
    )
    body = _published_body(orca, client, event)
    assert body == _system_event(
        "MLRun.LogCollector.Failed", LOG_COLLECTOR_DESCRIPTION, details={}
    )


def test_log_collector_renders_context(orca, client):
    event = client.generate_log_collector_event(
        mlrun.common.schemas.LogCollectorEventActions.failed,
        error=RuntimeError("collector unreachable"),
        run_uid="run-7",
        project="proj-a",
    )
    body = _published_body(orca, client, event)
    assert body == _system_event(
        "MLRun.LogCollector.Failed",
        LOG_COLLECTOR_DESCRIPTION,
        details={
            "run_uid": "run-7",
            "project": "proj-a",
            "error": "collector unreachable",
            "error_type": "RuntimeError",
        },
    )


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,config_name,description",
    [
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded,
            "MLRun.Project.Creation.Succeeded",
            "Project was successfully created",
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed,
            "MLRun.Project.Creation.Failed",
            "Project creation failed",
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_succeeded,
            "MLRun.Project.Deletion.Succeeded",
            "Project was successfully deleted",
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
            "MLRun.Project.Deletion.Failed",
            "Project deletion failed",
        ),
    ],
)
def test_project_lifecycle_event_payload(
    orca, client, action, config_name, description
):
    event = client.generate_project_lifecycle_event(
        action=action, project_name="my-project", actor="alice"
    )
    body = _published_body(orca, client, event)
    assert body == _system_event(
        config_name,
        description,
        details={"project_name": "my-project", "actor": "alice"},
    )


def test_project_lifecycle_omits_actor_when_missing(orca, client):
    event = client.generate_project_lifecycle_event(
        action=mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded,
        project_name="p",
        actor=None,
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {"project_name": "p"}


@pytest.mark.parametrize(
    "failed_action,description",
    [
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.creation_failed,
            "Project creation failed",
        ),
        (
            mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
            "Project deletion failed",
        ),
    ],
)
def test_project_lifecycle_failed_carries_error(
    orca, client, failed_action, description
):
    event = client.generate_project_lifecycle_event(
        action=failed_action,
        project_name="my-project",
        actor="alice",
        error=RuntimeError("db unavailable"),
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {
        "project_name": "my-project",
        "actor": "alice",
        "error": "db unavailable",
        "error_type": "RuntimeError",
    }
    assert body["description"] == description


def test_project_lifecycle_succeeded_ignores_error(orca, client):
    event = client.generate_project_lifecycle_event(
        action=mlrun.common.schemas.ProjectLifecycleEventActions.creation_succeeded,
        project_name="my-project",
        actor="alice",
        error=RuntimeError("should be ignored"),
    )
    body = _published_body(orca, client, event)
    assert body["details"] == {"project_name": "my-project", "actor": "alice"}
    assert body["description"] == "Project was successfully created"


# ---------------------------------------------------------------------------
# Audit events: project secret, permission denied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,config_name,description",
    [
        (
            mlrun.common.schemas.SecretEventActions.created,
            "UserAction.Project.MLRunSecret.Created",
            "Project my-project secret created",
        ),
        (
            mlrun.common.schemas.SecretEventActions.updated,
            "UserAction.Project.MLRunSecret.Updated",
            "Project my-project secret updated",
        ),
        (
            mlrun.common.schemas.SecretEventActions.deleted,
            "UserAction.Project.MLRunSecret.Deleted",
            "Project my-project secret deleted",
        ),
    ],
)
@pytest.mark.parametrize("initiator", ["alice", None])
def test_project_secret_event_payload(
    orca, client, action, config_name, description, initiator
):
    event = client.generate_project_secret_event(
        project="my-project",
        secret_name="mlrun.secret.k8s",
        secret_keys=["key"],
        action=action,
        initiator=initiator,
    )
    body = _published_body(orca, client, event)
    expected = {
        "class": "user action",
        "configName": config_name,
        "description": description,
        "details": {},
        "entityName": "my-project",
        "kind": "audit",
        "projectName": "my-project",
        "severity": "unspecified",
        "source": "",
        "traceId": "",
    }
    # An unknown initiator is left out of the body, not sent as null
    if initiator is not None:
        expected["initiator"] = initiator
    assert body == expected


def test_permission_denied_event_payload(orca, client):
    event = client.generate_permission_denied_event(
        resource="/projects/my-project/functions/my-function",
        action="create",
        username="alice",
    )
    body = _published_body(orca, client, event)
    assert body == {
        "class": "user action",
        "configName": "UserAction.General.PermissionDenied",
        "description": (
            "Permission denied for user alice to perform action create in resource "
            "/projects/my-project/functions/my-function"
        ),
        "details": {
            "username": "alice",
            "action": "create",
            "resource": "/projects/my-project/functions/my-function",
        },
        "entityName": "/projects/my-project/functions/my-function",
        "initiator": "alice",
        "kind": "audit",
        "projectName": "",
        "severity": "unspecified",
        "source": "",
        "traceId": "",
    }


def test_permission_denied_event_no_username(orca, client):
    event = client.generate_permission_denied_event(
        resource="/projects/my-project/functions/my-function",
        action="read",
    )
    body = _published_body(orca, client, event)
    assert "initiator" not in body
    # details carry the username as an explicit null
    assert body["details"] == {
        "username": None,
        "action": "read",
        "resource": "/projects/my-project/functions/my-function",
    }
    assert body["description"] == (
        "Permission denied for user None to perform action read in resource "
        "/projects/my-project/functions/my-function"
    )


# ---------------------------------------------------------------------------
# Behavior shared by the generators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generate,description",
    [
        (
            lambda c, error: c.generate_db_migration_event(
                mlrun.common.schemas.MigrationEventActions.failed, error=error
            ),
            MIGRATION_FAILED_DESCRIPTION,
        ),
        (
            lambda c, error: c.generate_db_connection_event(
                mlrun.common.schemas.DBConnectionEventActions.failed, error=error
            ),
            DB_CONNECTION_DESCRIPTION,
        ),
        (
            lambda c, error: c.generate_log_collector_event(
                mlrun.common.schemas.LogCollectorEventActions.failed, error=error
            ),
            LOG_COLLECTOR_DESCRIPTION,
        ),
        (
            lambda c, error: c.generate_project_lifecycle_event(
                action=mlrun.common.schemas.ProjectLifecycleEventActions.deletion_failed,
                project_name="my-project",
                error=error,
            ),
            "Project deletion failed",
        ),
    ],
    ids=["db_migration", "db_connection", "log_collector", "project_lifecycle"],
)
def test_error_detail_truncated(orca, client, generate, description):
    body = _published_body(orca, client, generate(client, LONG_ERROR))
    assert body["details"]["error"] == TRUNCATED_ERROR
    assert body["description"] == description


@pytest.mark.parametrize(
    "generate",
    [
        lambda c: c.generate_db_migration_event("bogus"),
        lambda c: c.generate_db_connection_event("bogus"),
        lambda c: c.generate_log_collector_event("bogus"),
        lambda c: c.generate_project_lifecycle_event(
            action="bogus", project_name="my-project"
        ),
        lambda c: c.generate_project_secret_event(
            project="my-project", secret_name="mlrun.secret.k8s", action="bogus"
        ),
    ],
    ids=[
        "db_migration",
        "db_connection",
        "log_collector",
        "project_lifecycle",
        "project_secret",
    ],
)
def test_unsupported_action_raises(orca, client, generate):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        generate(client)
    assert orca.log == []


def test_auth_secret_event_is_noop(orca, client):
    event = client.generate_auth_secret_event(
        username="alice",
        secret_name="secret",
        action=mlrun.common.schemas.AuthSecretEventActions.created,
    )
    assert event is None
    client.emit(event)
    assert orca.log == []


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------


def test_emit_request_contract(orca, client):
    orca.expect_oneshot_request(EVENTS_PATH, method="POST").respond_with_json(
        ACTIVATION_RESPONSE, status=201
    )
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    assert len(orca.log) == 1
    request = orca.log[0][0]
    assert request.method == "POST"
    assert request.path == EVENTS_PATH
    assert request.query_string == b""
    # Events are published with the API server's service-account identity
    assert request.headers["Authorization"] == f"Bearer {SA_TOKEN}"
    assert request.headers["X-Igz-Authenticator-Kind"] == "sa"
    assert request.headers["Content-Type"] == "application/json"


@pytest.mark.parametrize("with_context_id", [True, False], ids=["set", "unset"])
def test_emit_propagates_context_id(orca, client, request, with_context_id):
    if with_context_id:
        request.getfixturevalue("context_id")
    orca.expect_oneshot_request(EVENTS_PATH, method="POST").respond_with_json(
        ACTIVATION_RESPONSE, status=201
    )
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    headers = orca.log[0][0].headers
    if with_context_id:
        assert headers["x-igz-ctx"] == CTX_ID
    else:
        assert "x-igz-ctx" not in headers


@pytest.mark.parametrize("status", [200, 201])
def test_emit_success_logs_no_warning(orca, client, warning_log, status):
    orca.expect_oneshot_request(EVENTS_PATH, method="POST").respond_with_json(
        ACTIVATION_RESPONSE, status=status
    )
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    assert len(orca.log) == 1
    warning_log.assert_not_called()


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_emit_error_status_swallowed_without_retry(orca, client, warning_log, status):
    orca.expect_request(EVENTS_PATH, method="POST").respond_with_json(
        {"status": {"errorMessage": "rejected"}}, status=status
    )
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    # Publishing is best-effort: the failure is logged, not raised, and not retried
    assert len(orca.log) == 1
    warning_log.assert_called_once()
    assert warning_log.call_args.kwargs["config_name"] == "MLRun.DB.Migration.Started"


def test_emit_connection_failure_swallowed(unreachable_orca, warning_log):
    client = _make_client()
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    warning_log.assert_called_once()


def test_emit_failure_log_omits_token(orca, client, warning_log):
    orca.expect_request(EVENTS_PATH, method="POST").respond_with_json({}, status=500)
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    warning_log.assert_called_once()
    assert SA_TOKEN not in str(warning_log.call_args)


def test_emit_none_sends_nothing(orca, client):
    client.emit(None)
    assert orca.log == []


@pytest.mark.parametrize("status", [202, 204])
def test_emit_accepts_non_error_2xx(orca, client, warning_log, status):
    orca.expect_oneshot_request(EVENTS_PATH, method="POST").respond_with_data(
        "", status=status
    )
    client.emit(
        client.generate_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.started
        )
    )
    assert len(orca.log) == 1
    warning_log.assert_not_called()


@pytest.mark.parametrize("ssl_verify", [True, False])
def test_emit_request_options(client, ssl_verify):
    mlrun.mlconf.iguazio_api_ssl_verify = ssl_verify
    with unittest.mock.patch.object(requests.Session, "post", autospec=True) as post:
        post.return_value.status_code = 201
        client.emit(
            client.generate_db_migration_event(
                mlrun.common.schemas.MigrationEventActions.started
            )
        )
    post.assert_called_once()
    assert post.call_args.kwargs["verify"] is ssl_verify
    assert post.call_args.kwargs["timeout"] == 60
    # A redirect is not followed, so the event is never re-sent elsewhere
    assert post.call_args.kwargs["allow_redirects"] is False


def test_emit_retries_failed_connections(unreachable_orca, warning_log):
    client = _make_client()
    with (
        unittest.mock.patch.object(
            urllib3.util.Retry, "get_backoff_time", return_value=0
        ),
        unittest.mock.patch.object(
            urllib3.connection.HTTPConnection,
            "_new_conn",
            autospec=True,
            side_effect=urllib3.connection.HTTPConnection._new_conn,
        ) as new_conn,
    ):
        client.emit(
            client.generate_db_migration_event(
                mlrun.common.schemas.MigrationEventActions.started
            )
        )
    # The first attempt plus three retries
    assert new_conn.call_count == 4
    warning_log.assert_called_once()


def _make_client(
    service_name: str = "api", role: str = "chief"
) -> iguazio_v4_events.Client:
    mlrun.mlconf.services.service_name = service_name
    mlrun.mlconf.httpdb.clusterization.role = role
    return iguazio_v4_events.Client()


def _published_body(orca: pytest_httpserver.HTTPServer, client, event) -> dict:
    orca.expect_oneshot_request(EVENTS_PATH, method="POST").respond_with_json(
        ACTIVATION_RESPONSE, status=201
    )
    client.emit(event)
    assert len(orca.log) == 1
    return orca.log[0][0].get_json()


def _system_event(
    config_name: str,
    description: str,
    details: dict,
    entity_name: str = "mlrun-api-chief",
) -> dict:
    """System events leave kind, class and severity unset so Orca's catalog fills them in."""
    return {
        "configName": config_name,
        "description": description,
        "details": details,
        "entityName": entity_name,
        "kind": "",
        "projectName": "",
        "severity": "unspecified",
        "source": "",
        "traceId": "",
    }
