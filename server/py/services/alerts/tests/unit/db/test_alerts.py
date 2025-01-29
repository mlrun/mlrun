# Copyright 2024 Iguazio
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

from datetime import datetime, timezone

import mlrun.common.schemas.alert as alert_objects

import services.alerts.tests.unit.crud.utils
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestAlerts(TestDatabaseBase):
    def test_store_alert_created_time(self):
        project = "project"
        alert_name = "test-alert"
        alert_entity = alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
            project=project,
            ids=[1234],
        )
        alert_summary = "testing 1 2 3"
        event_kind = alert_objects.EventKind.DATA_DRIFT_DETECTED

        alert1 = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            summary=alert_summary,
            event_kind=event_kind,
        )

        self._db.store_alert(self._db_session, alert1)
        alerts = self._db.list_alerts(self._db_session, project)
        assert len(alerts) == 1

        assert alerts[0].created.replace(tzinfo=timezone.utc) < datetime.now(
            tz=timezone.utc
        )

        alert2_name = "test-alert2"
        alert2 = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert2_name,
            entity=alert_entity,
            summary=alert_summary,
            event_kind=event_kind,
        )

        self._db.store_alert(self._db_session, alert2)
        alerts = self._db.list_alerts(self._db_session, project)
        assert len(alerts) == 2
        alert1_created_time = alerts[0].created
        alert2_created_time = alerts[1].created

        assert alert1_created_time < alert2_created_time
