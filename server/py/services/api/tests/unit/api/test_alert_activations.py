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

from fastapi.testclient import TestClient

import mlrun.common.schemas

import services.api.crud
import services.api.tests.unit.api.utils


@unittest.mock.patch.object(services.api.crud.AlertActivation, "list_alert_activations")
def test_list_alert_activations(patched_list_alert_activations, client: TestClient):
    alert_name = "alert-name"
    project_name = "project-name"

    services.api.tests.unit.api.utils.create_project(client, project_name)
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

    assert result_from_global_endpoint.json() == result_from_alert_name_endpoint.json()
