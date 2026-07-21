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

import asyncio
import collections.abc
import unittest.mock

import fastapi
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun

import framework.middlewares
import framework.middlewares.base
import framework.middlewares.rest_metrics
import framework.utils.telemetry.rest_metrics
import services.api.main


def test_is_response_start_matches_only_response_start_message() -> None:
    assert (
        framework.middlewares.base.is_response_start(
            {"type": "http.response.start", "status": 200, "headers": []}
        )
        is True
    )
    assert (
        framework.middlewares.base.is_response_start(
            {"type": "http.response.body", "body": b"", "more_body": False}
        )
        is False
    )
    assert (
        framework.middlewares.base.is_response_start({"type": "http.request"}) is False
    )


@pytest.mark.parametrize(
    "path,expected",
    [
        # project-scoped: resource is the segment after {project}
        ("/api/v1/projects/proj/functions/fn", ("functions", "proj")),
        ("/api/v1/projects/proj/runs/uid-123", ("runs", "proj")),
        ("/api/v1/projects/proj/artifacts/key/with/slashes", ("artifacts", "proj")),
        # operating on the project itself
        ("/api/v1/projects/proj", ("projects", "proj")),
        ("/api/v1/projects", ("projects", "")),
        # non-project routes: resource is the first segment
        ("/api/v1/runs", ("runs", "")),
        ("/api/v1/client-spec", ("client-spec", "")),
        # v2 + legacy unversioned mounts
        ("/api/v2/projects/proj/artifacts", ("artifacts", "proj")),
        ("/api/projects/proj/functions/fn", ("functions", "proj")),
        # nothing to parse
        ("/", ("", "")),
        ("/api/v1", ("", "")),
    ],
)
def test_parse_resource_and_project(path: str, expected: tuple[str, str]) -> None:
    assert framework.middlewares.rest_metrics.parse_resource_and_project(path) == (
        expected
    )


def _middleware_class_names(app: fastapi.FastAPI) -> list[str]:
    return [middleware.cls.__name__ for middleware in app.user_middleware]


@pytest.fixture
def service() -> services.api.main.Service:
    svc = services.api.main.Service()
    svc._initialize_app()
    return svc


@pytest.fixture(autouse=True)
def _reset_telemetry_config() -> collections.abc.Iterator[None]:
    original_enabled = mlrun.mlconf.telemetry.enabled
    original_rest_metrics_enabled = mlrun.mlconf.telemetry.rest_metrics.enabled
    original_otlp_endpoint = mlrun.mlconf.telemetry.otlp_endpoint
    yield
    mlrun.mlconf.telemetry.enabled = original_enabled
    mlrun.mlconf.telemetry.rest_metrics.enabled = original_rest_metrics_enabled
    mlrun.mlconf.telemetry.otlp_endpoint = original_otlp_endpoint


def test_rest_metrics_config_defaults() -> None:
    assert mlrun.mlconf.telemetry.enabled is False
    assert mlrun.mlconf.telemetry.rest_metrics.enabled is True


def test_rest_metrics_middleware_absent_by_default(
    service: services.api.main.Service,
) -> None:
    service._add_middlewares()
    assert "RestMetricsMiddleware" not in _middleware_class_names(service.app)


def test_rest_metrics_middleware_absent_when_master_switch_off(
    service: services.api.main.Service,
) -> None:
    mlrun.mlconf.telemetry.enabled = False
    mlrun.mlconf.telemetry.rest_metrics.enabled = True
    service._add_middlewares()
    assert "RestMetricsMiddleware" not in _middleware_class_names(service.app)


def test_rest_metrics_middleware_absent_when_sub_flag_off(
    service: services.api.main.Service,
) -> None:
    mlrun.mlconf.telemetry.enabled = True
    mlrun.mlconf.telemetry.rest_metrics.enabled = False
    service._add_middlewares()
    assert "RestMetricsMiddleware" not in _middleware_class_names(service.app)


def test_rest_metrics_middleware_registered_when_enabled(
    service: services.api.main.Service,
) -> None:
    mlrun.mlconf.telemetry.enabled = True
    mlrun.mlconf.telemetry.rest_metrics.enabled = True
    mlrun.mlconf.telemetry.otlp_endpoint = "http://otel-collector:4317"
    service._add_middlewares()
    assert "RestMetricsMiddleware" in _middleware_class_names(service.app)


async def _run_middleware(app, scope):
    sent_messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent_messages.append(message)

    await framework.middlewares.RestMetricsMiddleware(app)(scope, receive, send)
    return sent_messages


def _http_scope(path: str = "/api/v1/projects/proj/functions/fn") -> dict:
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [],
    }


def test_rest_metrics_middleware_records_duration_on_response_start() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with unittest.mock.patch.object(
        framework.utils.telemetry.rest_metrics, "record_duration"
    ) as record_duration:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    record_duration.assert_called_once()
    kwargs = record_duration.call_args.kwargs
    assert kwargs["method"] == "GET"
    assert kwargs["status_code"] == 200
    assert kwargs["duration_ms"] >= 0
    assert kwargs["resource"] == "functions"
    assert kwargs["project"] == "proj"


def test_rest_metrics_middleware_excludes_healthz() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with unittest.mock.patch.object(
        framework.utils.telemetry.rest_metrics, "record_duration"
    ) as record_duration:
        asyncio.run(_run_middleware(downstream_app, _http_scope(path="/api/healthz")))

    record_duration.assert_not_called()


def test_rest_metrics_middleware_records_once_for_streamed_body() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        await send({"type": "http.response.body", "body": b"b", "more_body": False})

    with unittest.mock.patch.object(
        framework.utils.telemetry.rest_metrics, "record_duration"
    ) as record_duration:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    record_duration.assert_called_once()


def test_rest_metrics_middleware_propagates_downstream_exceptions() -> None:
    async def failing_app(scope, receive, send):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(_run_middleware(failing_app, _http_scope()))


def test_rest_metrics_middleware_swallows_record_duration_exceptions() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    warning_mock = unittest.mock.MagicMock()
    with (
        unittest.mock.patch.object(
            framework.utils.telemetry.rest_metrics,
            "record_duration",
            side_effect=RuntimeError("instrument broken"),
        ),
        unittest.mock.patch.object(mlrun.utils.logger, "warning", warning_mock),
    ):
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    warning_mock.assert_called_once()
    assert "REST metrics recording failed" in warning_mock.call_args.args[0]


def test_rest_metrics_middleware_skips_non_http_scope() -> None:
    called = {"ran": False}

    async def app(scope, receive, send):
        called["ran"] = True

    with unittest.mock.patch.object(
        framework.utils.telemetry.rest_metrics, "record_duration"
    ) as record_duration:
        asyncio.run(_run_middleware(app, {"type": "lifespan"}))

    assert called["ran"] is True
    record_duration.assert_not_called()


def test_service_lifecycle_wires_rest_metrics_init_and_shutdown(
    db: sqlalchemy.orm.Session,
    app: fastapi.FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared service startup/teardown must call the telemetry lifecycle hooks.

    Guards the wiring in framework.service._setup_service / _teardown_service so
    the histogram provider is stood up (and flushed) on every replica.
    """
    mlrun.mlconf.telemetry.enabled = True
    mlrun.mlconf.telemetry.rest_metrics.enabled = True
    mlrun.mlconf.telemetry.otlp_endpoint = "http://otel-collector:4317"

    init_mock = unittest.mock.MagicMock()
    shutdown_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(framework.utils.telemetry.rest_metrics, "init", init_mock)
    monkeypatch.setattr(
        framework.utils.telemetry.rest_metrics, "shutdown", shutdown_mock
    )

    with fastapi.testclient.TestClient(app):
        pass

    init_mock.assert_called()
    shutdown_mock.assert_called()
