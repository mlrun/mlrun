# Copyright 2023 Iguazio
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

import datetime

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
from mlrun.utils import logger

import services.alerts.crud


class Events(
    metaclass=mlrun.utils.singleton.Singleton,
):
    # Exact-match cache: (project, event_kind, entity_id) -> set[alert_id]
    _cache: dict[tuple[str, str, str], set[int]] = {}
    # Wildcard cache for alerts with ids=["*"]: (project, event_kind) -> set[alert_id]
    _wildcard_cache: dict[tuple[str, str], set[int]] = {}
    cache_initialized = False

    @staticmethod
    def is_valid_event(project: str, event_data: mlrun.common.schemas.Event):
        if event_data.entity.project != project:
            return False

        return False if len(event_data.entity.ids) > 1 else bool(event_data.is_valid())

    def add_event_configuration(self, project, event_kind, alert_id, entity_id):
        """Register an alert to be triggered by a specific event kind and entity.

        When ``entity_id`` is ``"*"``, the alert matches events from *any*
        entity of the given kind (stored in ``_wildcard_cache``).  Otherwise,
        it matches only events with the exact entity id (``_cache``).
        """
        if entity_id == "*":
            self._wildcard_cache.setdefault((project, event_kind), set()).add(alert_id)
        else:
            self._cache.setdefault((project, event_kind, entity_id), set()).add(
                alert_id
            )

    def remove_event_configuration(self, project, event_kind, alert_id, entity_id):
        if entity_id == "*":
            key = (project, event_kind)
            alerts = self._wildcard_cache.get(key, set())
            if alert_id in alerts:
                alerts.remove(alert_id)
                if not alerts:
                    self._wildcard_cache.pop(key)
        else:
            key = (project, event_kind, entity_id)
            alerts = self._cache.get(key, set())
            if alert_id in alerts:
                alerts.remove(alert_id)
                if not alerts:
                    self._cache.pop(key)

    def delete_project_alert_events(self, project):
        self._cache = {
            key: value for key, value in self._cache.items() if key[0] != project
        }
        self._wildcard_cache = {
            key: value
            for key, value in self._wildcard_cache.items()
            if key[0] != project
        }

    def process_event(
        self,
        session: sqlalchemy.orm.Session,
        event_data: mlrun.common.schemas.Event,
        event_name: str,
        project: str | None = None,
        validate_event: bool = False,
    ):
        if validate_event and (
            project is None or not self.is_valid_event(project, event_data)
        ):
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid event specified {event_name}"
            )

        event_data.timestamp = datetime.datetime.now(datetime.UTC)

        if not self.cache_initialized:
            services.alerts.crud.Alerts().process_event_no_cache(
                session, event_name, event_data
            )
            return

        if project is None:
            return

        try:
            entity_id = event_data.entity.ids[0]
            exact_alerts = self._cache.get((project, event_name, entity_id), set())
            wildcard_alerts = self._wildcard_cache.get((project, event_name), set())

            # TODO: Remove log once the flow is stable
            logger.debug(
                "Processing alerts for event",
                project=project,
                event_name=event_name,
                entity=entity_id,
                num_of_alerts=len(exact_alerts) + len(wildcard_alerts),
            )
            for alert_id in exact_alerts:
                services.alerts.crud.Alerts().process_event(
                    session, alert_id, event_data
                )
            for alert_id in wildcard_alerts:
                services.alerts.crud.Alerts().process_event(
                    session, alert_id, event_data
                )
        except KeyError:
            logger.debug(
                "Received event has no associated alert",
                project=project,
                name=event_name,
            )
            return
