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

import framework.utils.auth.providers.opa as opa
import services.api.utils.events.permission_denied as permission_denied


def _factory_returning(client):
    return unittest.mock.MagicMock(return_value=client)


def test_publish_emits_event_via_factory(monkeypatch):
    fake_event = object()
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_permission_denied_event.return_value = fake_event
    monkeypatch.setattr(
        permission_denied.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(fake_client),
    )

    emitted = permission_denied.publish_permission_denied(
        resource="/projects/proj-a/functions/func-1",
        action="create",
        username="alice",
    )

    assert emitted is True
    fake_client.generate_permission_denied_event.assert_called_once_with(
        resource="/projects/proj-a/functions/func-1",
        action="create",
        username="alice",
    )
    fake_client.emit.assert_called_once_with(fake_event)


def test_publish_no_event_from_nop_client(monkeypatch):
    """A NopClient returns None from generate_*: emit is skipped and publish
    reports that nothing was emitted."""
    nop_client = unittest.mock.MagicMock()
    nop_client.generate_permission_denied_event.return_value = None
    monkeypatch.setattr(
        permission_denied.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(nop_client),
    )

    assert (
        permission_denied.publish_permission_denied(
            resource="/projects/proj-a/functions/func-1", action="read"
        )
        is False
    )
    nop_client.emit.assert_not_called()


def test_publish_returns_false_when_emit_raises(monkeypatch):
    """A raising emit (e.g. events service unreachable) is swallowed and
    reported as not-emitted; it must never propagate into the caller."""
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_permission_denied_event.return_value = object()
    fake_client.emit.side_effect = RuntimeError("events down")
    monkeypatch.setattr(
        permission_denied.events_factory.EventsFactory,
        "get_events_client",
        _factory_returning(fake_client),
    )

    assert (
        permission_denied.publish_permission_denied(
            resource="/projects/proj-a/functions/func-1", action="read"
        )
        is False
    )


def test_publish_swallows_factory_exception(monkeypatch):
    monkeypatch.setattr(
        permission_denied.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(side_effect=RuntimeError("network down")),
    )

    emitted = permission_denied.publish_permission_denied(
        resource="/projects/proj-a/functions/func-1", action="read"
    )
    assert emitted is False


def test_register_for_opa_provider_sets_opa_listener(monkeypatch):
    monkeypatch.setattr(opa, "_permission_denied_listener", None)

    permission_denied.register_for_opa_provider()

    assert (
        opa._permission_denied_listener is permission_denied.publish_permission_denied
    )
