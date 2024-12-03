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
import datetime
import unittest
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.common.schemas

import framework.utils.singletons.db
import services.alerts.crud
import services.api.tests.unit.api.utils
from services.alerts.tests.unit.conftest import TestAlertsBase

ALERTS_PATH = "projects/{project}/alerts"
STORE_ALERTS_PATH = "projects/{project}/alerts/{name}"


class TestAlerts(TestAlertsBase):
    def test_store_alerts(self, db: Session, client: TestClient):
        project = "test-alerts"
        alert_name = "alert-name"
        self._create_project(db, project)
        notification = mlrun.model.Notification(
            kind="slack",
            when=["completed", "error"],
            name="test-alert-notification",
            message="test-message",
            condition="",
            severity="info",
            params={"webhook": "some-value"},
        )
        alert_config = mlrun.common.schemas.AlertConfig(
            project=project,
            name=alert_name,
            summary="oops",
            severity=mlrun.common.schemas.alert.AlertSeverity.HIGH,
            entities={
                "kind": mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
                "project": project,
                "ids": [1234],
            },
            trigger={
                "events": [mlrun.common.schemas.alert.EventKind.DATA_DRIFT_DETECTED]
            },
            notifications=[{"notification": notification.to_dict()}],
            reset_policy=mlrun.common.schemas.alert.ResetPolicy.MANUAL,
        )
        resp = client.put(
            STORE_ALERTS_PATH.format(project=project, name=alert_name),
            json=alert_config.dict(),
        )
        assert resp.status_code == HTTPStatus.OK.value

        resp = client.get(
            ALERTS_PATH.format(project=project),
        )
        assert resp.status_code == HTTPStatus.OK.value
        alerts = resp.json()
        assert len(alerts) == 1
        assert alerts[0]["name"] == alert_name

    @unittest.mock.patch.object(
        services.alerts.crud.AlertActivation, "list_alert_activations"
    )
    def test_list_alert_activations(
        self, patched_list_alert_activations, db: Session, client: TestClient
    ):
        alert_name = "alert-name"
        project_name = "project-name"

        self._create_project(db, project_name)
        patched_list_alert_activations.return_value = [
            mlrun.common.schemas.AlertActivation(
                id=1,
                name=alert_name,
                project=project_name,
                severity=mlrun.common.schemas.alert.AlertSeverity.HIGH,
                activation_time=datetime.datetime.utcnow(),
                entity_id="1234",
                entity_kind=mlrun.common.schemas.alert.EventEntityKind.JOB,
                event_kind=mlrun.common.schemas.alert.EventKind.DATA_DRIFT_SUSPECTED,
                number_of_events=1,
                notifications=[],
                criteria=mlrun.common.schemas.alert.AlertCriteria(count=1),
            )
        ]
        # to appear in the methods which allow pagination
        patched_list_alert_activations.__name__ = "list_alert_activations"

        result_from_global_endpoint = client.get(
            f"projects/{project_name}/alert-activations"
        )
        assert result_from_global_endpoint.status_code == 200

        result_from_alert_name_endpoint = client.get(
            f"projects/{project_name}/alerts/{alert_name}/activations"
        )
        assert result_from_alert_name_endpoint.status_code == 200

        assert (
            result_from_global_endpoint.json() == result_from_alert_name_endpoint.json()
        )

    # TODO: Move to test utils framework
    def _create_project(self, session: Session, project_name: str):
        db = framework.utils.singletons.db.get_db()
        db.create_project(
            session,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
            ),
        )
