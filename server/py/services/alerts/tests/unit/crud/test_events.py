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

import pytest

import mlrun.common.schemas.alert as alert_objects

import services.alerts.crud


@pytest.fixture()
def events():
    """Provide a fresh Events instance with cleared caches."""
    instance = services.alerts.crud.Events()
    instance._cache.clear()
    instance._wildcard_cache.clear()
    instance.cache_initialized = True
    yield instance
    instance._cache.clear()
    instance._wildcard_cache.clear()


PROJECT = "test-project"
EVENT_KIND = alert_objects.EventKind.MODEL_MONITORING_LAG_DETECTED


class TestAddEventConfiguration:
    def test_exact_entity_goes_to_exact_cache(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="ep1")

        assert 1 in events._cache[(PROJECT, EVENT_KIND, "ep1")]
        assert not events._wildcard_cache

    def test_wildcard_entity_goes_to_wildcard_cache(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="*")

        assert 1 in events._wildcard_cache[(PROJECT, EVENT_KIND)]
        assert not events._cache

    def test_multiple_alerts_same_key(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="*")
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=2, entity_id="*")

        assert events._wildcard_cache[(PROJECT, EVENT_KIND)] == {1, 2}


class TestRemoveEventConfiguration:
    def test_remove_exact(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="ep1")
        events.remove_event_configuration(
            PROJECT, EVENT_KIND, alert_id=1, entity_id="ep1"
        )

        assert not events._cache

    def test_remove_wildcard(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="*")
        events.remove_event_configuration(
            PROJECT, EVENT_KIND, alert_id=1, entity_id="*"
        )

        assert not events._wildcard_cache

    def test_remove_one_of_many(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="*")
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=2, entity_id="*")
        events.remove_event_configuration(
            PROJECT, EVENT_KIND, alert_id=1, entity_id="*"
        )

        assert events._wildcard_cache[(PROJECT, EVENT_KIND)] == {2}

    def test_remove_nonexistent_is_noop(self, events):
        events.remove_event_configuration(
            PROJECT, EVENT_KIND, alert_id=99, entity_id="*"
        )

        assert not events._wildcard_cache


class TestDeleteProjectAlertEvents:
    def test_clears_both_caches_for_project(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="ep1")
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=2, entity_id="*")
        events.add_event_configuration(
            "other-project", EVENT_KIND, alert_id=3, entity_id="ep2"
        )
        events.add_event_configuration(
            "other-project", EVENT_KIND, alert_id=4, entity_id="*"
        )

        events.delete_project_alert_events(PROJECT)

        assert all(k[0] != PROJECT for k in events._cache)
        assert all(k[0] != PROJECT for k in events._wildcard_cache)
        assert 3 in events._cache[("other-project", EVENT_KIND, "ep2")]
        assert 4 in events._wildcard_cache[("other-project", EVENT_KIND)]


class TestProcessEventWildcard:
    @staticmethod
    def _make_event(entity_id: str) -> alert_objects.Event:
        return alert_objects.Event(
            kind=EVENT_KIND,
            entity=alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.MODEL_MONITORING_INFRA,
                project=PROJECT,
                ids=[entity_id],
            ),
        )

    def test_wildcard_alert_matches_any_entity(self, events):
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=1, entity_id="*")
        session = unittest.mock.Mock()
        event = self._make_event("myproject.writer.3")

        with unittest.mock.patch.object(
            services.alerts.crud.Alerts, "process_event"
        ) as mock_process:
            events.process_event(session, event, EVENT_KIND, project=PROJECT)

        mock_process.assert_called_once_with(session, 1, event)

    def test_exact_alert_does_not_match_other_entity(self, events):
        events.add_event_configuration(
            PROJECT, EVENT_KIND, alert_id=1, entity_id="writer.0"
        )
        session = unittest.mock.Mock()
        event = self._make_event("writer.1")

        with unittest.mock.patch.object(
            services.alerts.crud.Alerts, "process_event"
        ) as mock_process:
            events.process_event(session, event, EVENT_KIND, project=PROJECT)

        mock_process.assert_not_called()

    def test_both_exact_and_wildcard_fire(self, events):
        events.add_event_configuration(
            PROJECT, EVENT_KIND, alert_id=1, entity_id="writer.0"
        )
        events.add_event_configuration(PROJECT, EVENT_KIND, alert_id=2, entity_id="*")
        session = unittest.mock.Mock()
        event = self._make_event("writer.0")

        with unittest.mock.patch.object(
            services.alerts.crud.Alerts, "process_event"
        ) as mock_process:
            events.process_event(session, event, EVENT_KIND, project=PROJECT)

        assert mock_process.call_count == 2
        fired_ids = {call.args[1] for call in mock_process.call_args_list}
        assert fired_ids == {1, 2}
