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
import http
import json
import types
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


def _route(endpoint_name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(endpoint=types.SimpleNamespace(__name__=endpoint_name))


@pytest.mark.parametrize(
    "method,route,expected",
    [
        # list_ prefix -> collection
        (
            http.HTTPMethod.GET,
            _route("list_runs"),
            framework.utils.telemetry.rest_metrics.GetVsList.LIST,
        ),
        (
            http.HTTPMethod.GET,
            _route("list_artifact_tags"),
            framework.utils.telemetry.rest_metrics.GetVsList.LIST,
        ),
        (
            http.HTTPMethod.GET,
            _route("list_pipelines"),
            framework.utils.telemetry.rest_metrics.GetVsList.LIST,
        ),
        (
            http.HTTPMethod.GET,
            _route("list_model_endpoints"),
            framework.utils.telemetry.rest_metrics.GetVsList.LIST,
        ),
        # explicit get_ prefix -> single identified object
        (
            http.HTTPMethod.GET,
            _route("get_run"),
            framework.utils.telemetry.rest_metrics.GetVsList.GET,
        ),
        (
            http.HTTPMethod.GET,
            _route("get_artifact"),
            framework.utils.telemetry.rest_metrics.GetVsList.GET,
        ),
        # singleton/action endpoints with no get_/list_ convention at all,
        # but which return a single object, not a collection
        (
            http.HTTPMethod.GET,
            _route("build_status"),
            framework.utils.telemetry.rest_metrics.GetVsList.GET,
        ),
        (
            http.HTTPMethod.GET,
            _route("clusterization_spec"),
            framework.utils.telemetry.rest_metrics.GetVsList.GET,
        ),
        (
            http.HTTPMethod.GET,
            _route("get_client_spec"),
            framework.utils.telemetry.rest_metrics.GetVsList.GET,
        ),
        # unmatched/404 routes never get a scope["route"]
        (http.HTTPMethod.GET, None, ""),
        # verb classification only applies to GET
        (http.HTTPMethod.POST, _route("list_runs"), ""),
        (http.HTTPMethod.DELETE, _route("get_run"), ""),
    ],
)
def test_parse_get_vs_list(
    method: str, route: types.SimpleNamespace | None, expected: str
) -> None:
    scope = {"route": route} if route is not None else {}
    assert framework.middlewares.rest_metrics.parse_get_vs_list(method, scope) == (
        expected
    )


@pytest.mark.parametrize(
    "body,expected",
    [
        (b"", None),
        (b"not json", None),
        (json.dumps([1, 2, 3]).encode(), 3),
        # list_runs/list_functions envelope: a sibling "pagination" key, but
        # pagination info (page_info: dict[str, str | int]) is never itself
        # list-valued, so it can't compete with the real collection field.
        (json.dumps({"runs": [1, 2], "pagination": {"page": 1}}).encode(), 2),
        (json.dumps({"funcs": []}).encode(), 0),
        # ModelEndpointList (schemas.ModelEndpointList) serializes the same
        # way: a single top-level list-valued field, just named differently.
        (json.dumps({"endpoints": [1, 2, 3]}).encode(), 3),
        (json.dumps({"pagination": {"page": 1}}).encode(), None),
        (json.dumps("just a string").encode(), None),
    ],
)
def test_parse_item_count(body: bytes, expected: int | None) -> None:
    assert framework.middlewares.rest_metrics.parse_item_count(body) == expected


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
    original_sample_rate = mlrun.mlconf.telemetry.rest_metrics.sample_rate
    yield
    mlrun.mlconf.telemetry.enabled = original_enabled
    mlrun.mlconf.telemetry.rest_metrics.enabled = original_rest_metrics_enabled
    mlrun.mlconf.telemetry.otlp_endpoint = original_otlp_endpoint
    mlrun.mlconf.telemetry.rest_metrics.sample_rate = original_sample_rate


def test_rest_metrics_config_defaults() -> None:
    assert mlrun.mlconf.telemetry.enabled is False
    assert mlrun.mlconf.telemetry.rest_metrics.enabled is True
    assert mlrun.mlconf.telemetry.rest_metrics.sample_rate == 1.0


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


def _make_receive(
    messages: collections.abc.Sequence[dict],
) -> collections.abc.Callable[[], collections.abc.Awaitable[dict]]:
    it = iter(messages)

    async def receive() -> dict:
        try:
            return next(it)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}

    return receive


async def _run_middleware(app, scope, receive=None):
    sent_messages = []

    if receive is None:
        receive = _make_receive([{"type": "http.request", "body": b""}])

    async def send(message):
        sent_messages.append(message)

    await framework.middlewares.RestMetricsMiddleware(app)(scope, receive, send)
    return sent_messages


def _http_scope(
    path: str = "/api/v1/projects/proj/functions/fn",
    method: str = http.HTTPMethod.GET,
    route_endpoint_name: str | None = None,
) -> dict:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [],
    }
    if route_endpoint_name is not None:
        scope["route"] = _route(route_endpoint_name)
    return scope


def _patch_all_record_fns() -> unittest.mock.patch.multiple:
    return unittest.mock.patch.multiple(
        framework.utils.telemetry.rest_metrics,
        record_duration=unittest.mock.DEFAULT,
        record_request_size=unittest.mock.DEFAULT,
        record_response_size=unittest.mock.DEFAULT,
        record_items_returned=unittest.mock.DEFAULT,
    )


def test_rest_metrics_middleware_records_after_response_completes() -> None:
    async def downstream_app(scope, receive, send):
        scope["route"] = _route("get_function")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    mocks["record_duration"].assert_called_once()
    kwargs = mocks["record_duration"].call_args.kwargs
    assert kwargs["method"] == http.HTTPMethod.GET
    assert kwargs["status_code"] == 200
    assert kwargs["duration_ms"] >= 0
    assert kwargs["resource"] == "functions"
    assert kwargs["project"] == "proj"
    assert kwargs["get_vs_list"] == framework.utils.telemetry.rest_metrics.GetVsList.GET


def test_rest_metrics_middleware_excludes_healthz() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope(path="/api/healthz")))

    mocks["record_duration"].assert_not_called()
    mocks["record_request_size"].assert_not_called()
    mocks["record_response_size"].assert_not_called()
    mocks["record_items_returned"].assert_not_called()


def test_rest_metrics_middleware_records_once_for_streamed_body() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        await send({"type": "http.response.body", "body": b"b", "more_body": False})

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    mocks["record_duration"].assert_called_once()


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

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(app, {"type": "lifespan"}))

    assert called["ran"] is True
    mocks["record_duration"].assert_not_called()


def test_rest_metrics_middleware_computes_request_size() -> None:
    async def downstream_app(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    receive = _make_receive(
        [
            {"type": "http.request", "body": b"01234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        ]
    )

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope(), receive=receive))

    mocks["record_request_size"].assert_called_once()
    request_size_kwargs = mocks["record_request_size"].call_args.kwargs
    assert request_size_kwargs["size_kib"] == pytest.approx(10 / 1024)


def test_rest_metrics_middleware_records_request_size_only_at_final_chunk() -> None:
    """Everything is recorded together once, at the final chunk — not before."""
    recorded_before_body_sent = {"value": None}

    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        recorded_before_body_sent["value"] = mocks["record_request_size"].called
        await send({"type": "http.response.body", "body": b"ok"})

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    assert recorded_before_body_sent["value"] is False
    mocks["record_request_size"].assert_called_once()


def test_rest_metrics_middleware_records_nothing_when_response_never_completes() -> (
    None
):
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        # Response body is never sent — the call hangs/never completes.

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    mocks["record_request_size"].assert_not_called()
    mocks["record_duration"].assert_not_called()


def test_rest_metrics_middleware_computes_response_size_across_chunks() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"a" * 512, "more_body": True}
        )
        await send(
            {"type": "http.response.body", "body": b"b" * 512, "more_body": False}
        )

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    response_size_kwargs = mocks["record_response_size"].call_args.kwargs
    assert response_size_kwargs["size_kib"] == pytest.approx(1024 / 1024)


def test_rest_metrics_middleware_records_item_count_for_list_calls() -> None:
    body = json.dumps({"runs": [{"a": 1}, {"a": 2}, {"a": 3}]}).encode()

    async def downstream_app(scope, receive, send):
        scope["route"] = _route("list_runs")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body})

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope(path="/api/v1/runs")))

    mocks["record_items_returned"].assert_called_once()
    assert mocks["record_items_returned"].call_args.kwargs["item_count"] == 3
    assert (
        mocks["record_duration"].call_args.kwargs["get_vs_list"]
        == framework.utils.telemetry.rest_metrics.GetVsList.LIST
    )


def test_rest_metrics_middleware_skips_item_count_for_get_calls() -> None:
    body = json.dumps({"run": {"a": 1}}).encode()

    async def downstream_app(scope, receive, send):
        scope["route"] = _route("get_run")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body})

    with _patch_all_record_fns() as mocks:
        asyncio.run(
            _run_middleware(downstream_app, _http_scope(path="/api/v1/runs/uid"))
        )

    mocks["record_items_returned"].assert_not_called()
    mocks["record_duration"].assert_called_once()


def test_rest_metrics_middleware_skips_all_recording_when_not_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        framework.utils.telemetry.rest_metrics, "should_sample", lambda **_: False
    )

    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with _patch_all_record_fns() as mocks:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    mocks["record_duration"].assert_not_called()
    mocks["record_request_size"].assert_not_called()
    mocks["record_response_size"].assert_not_called()
    mocks["record_items_returned"].assert_not_called()


def test_rest_metrics_middleware_passes_sampling_inputs() -> None:
    """should_sample is evaluated exactly once per call, from the complete
    picture (final chunk) — so a call is kept or dropped for every instrument
    together, never a mix.
    """

    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"x" * 2048})

    with unittest.mock.patch.object(
        framework.utils.telemetry.rest_metrics, "should_sample", return_value=True
    ) as should_sample:
        asyncio.run(_run_middleware(downstream_app, _http_scope()))

    should_sample.assert_called_once()
    kwargs = should_sample.call_args.kwargs
    assert kwargs["status_code"] == 404
    assert kwargs["response_size_kib"] == pytest.approx(2048 / 1024)
    assert kwargs["elapsed_seconds"] >= 0


def test_rest_metrics_middleware_swallows_recording_errors() -> None:
    async def downstream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with unittest.mock.patch.object(
        framework.utils.telemetry.rest_metrics,
        "record_duration",
        side_effect=RuntimeError("boom"),
    ):
        # Must not raise — a broken instrument can never fail the request.
        asyncio.run(_run_middleware(downstream_app, _http_scope()))


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
