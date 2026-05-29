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
import unittest.mock

import pytest

import mlrun
import mlrun.common.schemas

import framework.utils.clients.log_collector as log_collector_client
import services.api.utils.events.log_collector_errors as log_collector_errors


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(log_collector_errors._slot, "_last_emit_monotonic", 0.0)

    # `register_for_log_collector` constructs the LogCollectorClient singleton,
    # which imports a generated grpc stub module that requires `make schemas`;
    # stub out the proto wiring so the suite runs without generated protos.
    # Singleton instances themselves are wiped between tests by the autouse
    # `config_test_base` fixture in tests/common_fixtures.py, so we don't need
    # to pop them here.
    def _stub_proto_init(self):
        self._log_collector_pb2 = unittest.mock.MagicMock()
        self._log_collector_pb2_grpc = unittest.mock.MagicMock()

    monkeypatch.setattr(
        log_collector_client.LogCollectorClient,
        "_initialize_proto_client_imports",
        _stub_proto_init,
    )


def _factory_returning(client):
    return unittest.mock.MagicMock(return_value=client)


def test_publish_emits_event_via_factory(monkeypatch):
    fake_event = object()
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_log_collector_event.return_value = fake_event
    monkeypatch.setattr(
        log_collector_errors.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(fake_client),
    )

    emitted = log_collector_errors.publish_log_collector_failed(
        operation="get_logs",
        run_uid="run-1",
        project="proj-a",
        error=RuntimeError("collector unreachable"),
        error_code=1,
        error_category="get_logs_failed",
    )

    assert emitted is True
    fake_client.generate_log_collector_event.assert_called_once()
    call_kwargs = fake_client.generate_log_collector_event.call_args.kwargs
    assert call_kwargs["action"] == mlrun.common.schemas.LogCollectorEventActions.failed
    assert call_kwargs["operation"] == "get_logs"
    assert call_kwargs["run_uid"] == "run-1"
    assert call_kwargs["project"] == "proj-a"
    assert call_kwargs["error_category"] == "get_logs_failed"
    assert call_kwargs["error_code"] == 1
    fake_client.emit.assert_called_once_with(fake_event)


def test_publish_no_event_from_nop_client_does_not_consume_throttle(monkeypatch):
    """A NopClient returns None: emit is skipped AND the slot stays free."""
    nop_client = unittest.mock.MagicMock()
    nop_client.generate_log_collector_event.return_value = None
    real_client = unittest.mock.MagicMock()
    real_client.generate_log_collector_event.return_value = object()

    monkeypatch.setattr(
        log_collector_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(side_effect=[nop_client, real_client]),
    )

    assert (
        log_collector_errors.publish_log_collector_failed(
            operation="get_logs", error_category="get_logs_failed"
        )
        is False
    )
    nop_client.emit.assert_not_called()

    assert (
        log_collector_errors.publish_log_collector_failed(
            operation="get_logs", error_category="get_logs_failed"
        )
        is True
    )
    real_client.emit.assert_called_once()


def test_publish_throttled_within_interval(monkeypatch):
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_log_collector_event.return_value = object()
    monkeypatch.setattr(
        log_collector_errors.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(fake_client),
    )
    monkeypatch.setattr(
        mlrun.mlconf.events.log_collector, "min_emit_interval_seconds", 60
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(
        log_collector_errors.throttle.time, "monotonic", lambda: fake_now["value"]
    )

    assert (
        log_collector_errors.publish_log_collector_failed(operation="get_logs") is True
    )
    fake_now["value"] += 30
    assert (
        log_collector_errors.publish_log_collector_failed(operation="get_logs") is False
    )
    assert fake_client.emit.call_count == 1


def test_publish_unthrottled_after_interval(monkeypatch):
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_log_collector_event.return_value = object()
    monkeypatch.setattr(
        log_collector_errors.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(fake_client),
    )
    monkeypatch.setattr(
        mlrun.mlconf.events.log_collector, "min_emit_interval_seconds", 60
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(
        log_collector_errors.throttle.time, "monotonic", lambda: fake_now["value"]
    )

    assert (
        log_collector_errors.publish_log_collector_failed(operation="get_logs") is True
    )
    fake_now["value"] += 90  # past the throttle interval
    assert (
        log_collector_errors.publish_log_collector_failed(operation="get_logs") is True
    )
    assert fake_client.emit.call_count == 2


def test_publish_releases_slot_when_emit_raises(monkeypatch):
    """An emit that raises (events service unreachable) frees the slot so
    the next failure within the throttle window can retry delivery."""
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_log_collector_event.return_value = object()
    fake_client.emit.side_effect = [RuntimeError("events down"), None]
    monkeypatch.setattr(
        log_collector_errors.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(fake_client),
    )
    monkeypatch.setattr(
        mlrun.mlconf.events.log_collector, "min_emit_interval_seconds", 60
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(
        log_collector_errors.throttle.time, "monotonic", lambda: fake_now["value"]
    )

    assert (
        log_collector_errors.publish_log_collector_failed(operation="get_logs") is False
    )
    # No advance in time: slot was released, so the next attempt can claim.
    assert (
        log_collector_errors.publish_log_collector_failed(operation="get_logs") is True
    )
    assert fake_client.emit.call_count == 2


def test_publish_swallows_factory_exception(monkeypatch):
    monkeypatch.setattr(
        log_collector_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(side_effect=RuntimeError("network down")),
    )

    emitted = log_collector_errors.publish_log_collector_failed(
        operation="get_logs",
        error=RuntimeError("boom"),
    )
    assert emitted is False


def test_listener_dispatch_offloads_to_executor(monkeypatch):
    """
    The framework client fires `_notify_failure` from an `async def`. The
    listener must not run the synchronous HTTP publish inline (would block the
    event loop). With a running loop, the listener dispatches via
    `run_in_executor` and the publish never executes on the loop thread.
    """
    publish_called_inline = {"yes": False}

    def fake_publish(**_kwargs):
        publish_called_inline["yes"] = True

    monkeypatch.setattr(
        log_collector_errors, "publish_log_collector_failed", fake_publish
    )

    fake_loop = unittest.mock.MagicMock()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

    log_collector_errors._on_log_collector_failure(
        log_collector_client.LogCollectorFailureContext(
            operation="get_logs",
            error_category="get_logs_failed",
            run_uid="r",
            project="p",
        )
    )

    # Inline publish would block; dispatch must have gone through the executor.
    assert publish_called_inline["yes"] is False
    fake_loop.run_in_executor.assert_called_once()
    # First positional arg to run_in_executor is the executor (None == default).
    args, _kwargs = fake_loop.run_in_executor.call_args
    assert args[0] is None


def test_listener_falls_back_to_inline_when_no_running_loop(monkeypatch):
    """Outside an async context (e.g. unit tests calling the listener directly)
    the publish should still happen — inline, since there is no loop to defer
    to."""
    calls = []

    def fake_publish(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        log_collector_errors, "publish_log_collector_failed", fake_publish
    )

    def raises_no_loop():
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "get_running_loop", raises_no_loop)

    log_collector_errors._on_log_collector_failure(
        log_collector_client.LogCollectorFailureContext(
            operation="start_logs",
            error_category="start_logs_failed",
            run_uid="r",
            project="p",
        )
    )

    assert len(calls) == 1
    assert calls[0]["operation"] == "start_logs"
