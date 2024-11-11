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

from sqlalchemy.orm import Session

import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.notification

import services.api.tests.unit.crud.utils
from services.api.db.base import DBInterface


def test_store_alert_activation(db: DBInterface, db_session: Session):
    project = "project-name"
    alert_name = "failed-alert"
    alert_summary = "testing alert failure"
    alert_entity = alert_objects.EventEntities(
        kind=alert_objects.EventEntityKind.JOB,
        project=project,
        ids=["run-name"],
    )
    event_kind = alert_objects.EventKind.FAILED
    alert_criteria = alert_objects.AlertCriteria(count=5, period="10m")
    alert_timestamp = datetime(2024, 11, 5, 14, 30, 0).replace(tzinfo=timezone.utc)
    event_data = alert_objects.Event(
        kind=event_kind,
        entity=alert_entity,
        timestamp=alert_timestamp,
        value_dict={"uid": 123},
    )
    notifications_states = [
        mlrun.common.schemas.NotificationState(kind="git", err=""),
        mlrun.common.schemas.NotificationState(
            kind="slack", err="slack channel not found"
        ),
    ]

    alert_data = services.api.tests.unit.crud.utils.generate_alert_data(
        project=project,
        name=alert_name,
        entity=alert_entity,
        criteria=alert_criteria,
        summary=alert_summary,
        event_kind=event_kind,
    )
    alert_data.id = 111

    db.store_alert_activation(
        db_session,
        alert_data=alert_data,
        event_data=event_data,
        notifications_states=notifications_states,
    )

    stored_activations = db.list_alerts_activations(db_session, project=project)
    assert len(stored_activations) == 1
    assert stored_activations[0].name == alert_name
    assert stored_activations[0].project == project
    assert stored_activations[0].activation_time == alert_timestamp
    assert stored_activations[0].severity == alert_objects.AlertSeverity.LOW
    assert (
        stored_activations[0].entity_id
        == f"{alert_data.entities.ids[0]}.{event_data.value_dict.get('uid')}"
    )
    assert stored_activations[0].entity_kind == alert_entity.kind
    assert stored_activations[0].event_kind == event_kind
    assert stored_activations[0].number_of_events == alert_criteria.count
    assert stored_activations[0].notifications == notifications_states
