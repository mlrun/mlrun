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
import services.alerts.tests.unit.conftest
import services.alerts.tests.unit.crud.utils
import services.api.tests.unit.api.utils

ALERTS_PATH = "projects/{project}/alerts"
STORE_ALERTS_PATH = "projects/{project}/alerts/{name}"


class TestAlerts(services.alerts.tests.unit.conftest.TestAlertsBase):
    def test_store_alerts(self, db: Session, client: TestClient, k8s_secrets_mock):
        project = "test-alerts"
        alert_name = "alert-name"
        self._create_project(db, project)
        alert_config = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=services.alerts.tests.unit.crud.utils.generate_alert_entity(
                project=project
            ),
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

    def test_list_alerts_for_all_projects(
        self, db: Session, client: TestClient, k8s_secrets_mock
    ):
        alert_name = "alert-name"
        for i in range(2):
            project = f"test-alerts-{i}"
            self._create_project(db, project)
            alert_config = services.alerts.tests.unit.crud.utils.generate_alert_data(
                project=project,
                name=alert_name,
                entity=services.alerts.tests.unit.crud.utils.generate_alert_entity(
                    project=project
                ),
            )
            resp = client.put(
                STORE_ALERTS_PATH.format(project=project, name=alert_name),
                json=alert_config.dict(),
            )
            assert resp.status_code == HTTPStatus.OK.value

        # list alerts for all projects
        resp = client.get(
            ALERTS_PATH.format(project="*"),
        )
        assert resp.status_code == HTTPStatus.OK.value
        alerts = resp.json()
        assert len(alerts) == 2

        # list alerts for a specific project
        resp = client.get(
            ALERTS_PATH.format(project="test-alerts-0"),
        )
        assert resp.status_code == HTTPStatus.OK.value
        alerts = resp.json()
        assert len(alerts) == 1

        # list alerts for a non-existing project
        resp = client.get(
            ALERTS_PATH.format(project="non-existing-project"),
        )
        assert resp.status_code == HTTPStatus.NOT_FOUND.value
        assert "does not exist" in resp.text

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
    @staticmethod
    def _create_project(session: Session, project_name: str):
        db = framework.utils.singletons.db.get_db()
        db.create_project(
            session,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
            ),
        )
