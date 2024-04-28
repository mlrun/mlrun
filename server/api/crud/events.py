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

import sqlalchemy.orm

import mlrun.utils.singleton
import server.api.api.utils
import server.api.utils.singletons.db
from mlrun.utils import logger


class Events(
    metaclass=mlrun.utils.singleton.Singleton,
):
    # we cache alert names based on project and event name as key
    _cache: dict[(str, str), list[str]] = {}

    @staticmethod
    def is_valid_event(project: str, event_data: mlrun.common.schemas.Event):
        if event_data.entity.project != project:
            return False

        return bool(event_data.is_valid())

    def add_event_configuration(self, project, name, alert_name):
        self._cache.setdefault((project, name), []).append(alert_name)

    def remove_event_configuration(self, project, name):
        del self._cache[(project, name)]

    def delete_project_alert_events(self, project):
        to_delete = [name for proj, name in self._cache if proj == project]
        for name in to_delete:
            self.remove_event_configuration(project, name)

    def process_event(
        self,
        session: sqlalchemy.orm.Session,
        event_data: mlrun.common.schemas.Event,
        event_name: str,
        project: str = None,
        validate_event: bool = False,
    ):
        project = project or mlrun.mlconf.default_project

        if validate_event and not self.is_valid_event(project, event_data):
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid event specified {event_name}"
            )

        event_data.timestamp = datetime.datetime.now(datetime.timezone.utc)

        try:
            for name in self._cache[(project, event_name)]:
                server.api.crud.Alerts().process_event(
                    session, project, name, event_data
                )
        except KeyError:
            logger.warn("Received unknown event", project=project, name=event_name)
            return
