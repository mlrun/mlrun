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
#
import datetime
from typing import Optional

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
from mlrun.utils import logger

import services.alerts.crud


class Events(
    metaclass=mlrun.utils.singleton.Singleton,
):
    # we cache alert names based on project and event name as key
    # (project, name, entity_id) -> set[alert_id]
    # TODO: Rethink the cache structure once a single alert supports more than a single id
    _cache: dict[(str, str, str), set[int]] = {}
    cache_initialized = False

    @staticmethod
    def is_valid_event(project: str, event_data: mlrun.common.schemas.Event):
        if event_data.entity.project != project:
            return False

        if len(event_data.entity.ids) > 1:
            return False

        return bool(event_data.is_valid())

    def add_event_configuration(self, project, event_kind, alert_id, entity_id):
        self._cache.setdefault((project, event_kind, entity_id), set()).add(alert_id)

    def remove_event_configuration(self, project, event_kind, alert_id, entity_id):
        alerts = self._cache.get((project, event_kind, entity_id), set())
        if alert_id in alerts:
            alerts.remove(alert_id)
            if len(alerts) == 0:
                self._cache.pop((project, event_kind, entity_id))

    def delete_project_alert_events(self, project):
        to_delete = [key for key in self._cache if key[0] == project]
        for key in to_delete:
            self._cache.pop(key)

    def process_event(
        self,
        session: sqlalchemy.orm.Session,
        event_data: mlrun.common.schemas.Event,
        event_name: str,
        project: Optional[str] = None,
        validate_event: bool = False,
    ):
        project = project or mlrun.mlconf.default_project

        if validate_event and not self.is_valid_event(project, event_data):
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid event specified {event_name}"
            )

        event_data.timestamp = datetime.datetime.now(datetime.timezone.utc)

        if not self.cache_initialized:
            services.alerts.crud.Alerts().process_event_no_cache(
                session, event_name, event_data
            )
            return

        try:
            # TODO: Remove log once the flow is stable
            logger.debug(
                "Processing alerts for event",
                project=project,
                event_name=event_name,
                entity=event_data.entity.ids[0],
                num_of_alerts=len(
                    self._cache.get(
                        (project, event_name, event_data.entity.ids[0]), set()
                    )
                ),
            )
            for alert_id in self._cache.get(
                (project, event_name, event_data.entity.ids[0]), set()
            ):
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
