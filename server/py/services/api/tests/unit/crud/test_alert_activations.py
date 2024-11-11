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

import fastapi.concurrency
import sqlalchemy.orm

import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.notification

import services.api.crud.alert_activation
import services.api.tests.unit.conftest
import services.api.tests.unit.crud.utils

async def test_store_alert_activation(
    db: sqlalchemy.orm.Session,
    k8s_secrets_mock: services.api.tests.unit.conftest.K8sSecretsMock,
):
    project = "project-name"
    alert_name = "failed-alert"
    alert_summary = "testing alert failure"
    alert_entity = alert_objects.EventEntities(
        kind=alert_objects.EventEntityKind.JOB,
        project=project,
        ids=["run-name"],
    )
    event_kind = alert_objects.EventKind.FAILED
    event_data = alert_objects.Event(
        kind=event_kind, entity=alert_entity, value_dict={"uid": 123}
    )

    alert_data = services.api.tests.unit.crud.utils.generate_alert_data(
        project=project,
        name=alert_name,
        entity=alert_entity,
        summary=alert_summary,
        event_kind=event_kind,
    )

    services.api.crud.Alerts().store_alert(
        session=db,
        project=project,
        name=alert_name,
        alert_data=alert_data,
    )

    # activate the alert
    await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Events().process_event,
        session=db,
        event_data=event_data,
        event_name=event_data.kind,
        project=project,
    )

    # since the reset policy of the alert if auto, then now it should be reset and added to the alert
    # activation table
    activations = services.api.crud.AlertActivation().list_alerts_activations(
        session=db, project=project
    )
    expected_notifications_states = [
        mlrun.common.schemas.NotificationState(kind="slack", status=""),
    ]
    expected_entity_id = (
        f"{alert_data.entities.ids[0]}.{event_data.value_dict.get('uid')}"
    )
    expected_activation_time = event_data.timestamp

    # since criteria is not defined the default count number is 1
    expected_number_of_events = 1

    assert len(activations) == 1
    _assert_alert_activation(
        activations[0],
        alert_name=alert_name,
        project=project,
        entity_kind=alert_entity.kind,
        entity_id=expected_entity_id,
        notifications_states=expected_notifications_states,
        activation_time=expected_activation_time,
        number_of_events=expected_number_of_events,
    )

    # trigger the event again and validate that the activation is saved again in the db
    await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Events().process_event,
        session=db,
        event_data=event_data,
        event_name=event_data.kind,
        project=project,
    )

    activations = services.api.crud.AlertActivation().list_alerts_activations(
        session=db, project=project
    )
    expected_activation_time = event_data.timestamp

    assert len(activations) == 2
    _assert_alert_activation(
        activations[1],
        alert_name=alert_name,
        project=project,
        entity_kind=alert_entity.kind,
        entity_id=expected_entity_id,
        notifications_states=expected_notifications_states,
        activation_time=expected_activation_time,
        number_of_events=1,
    )


async def test_store_alert_activation_with_criteria(
    db: sqlalchemy.orm.Session,
    k8s_secrets_mock: services.api.tests.unit.conftest.K8sSecretsMock,
):
    project = "project-name"
    alert_name = "failed-alert"
    alert_summary = "testing alert failure"
    alert_entity = alert_objects.EventEntities(
        kind=alert_objects.EventEntityKind.JOB,
        project=project,
        ids=["run-name"],
    )
    event_kind = alert_objects.EventKind.FAILED
    event_data = alert_objects.Event(
        kind=event_kind, entity=alert_entity, value_dict={"uid": 123}
    )
    alert_criteria = alert_objects.AlertCriteria(count=3)

    alert_data = services.api.tests.unit.crud.utils.generate_alert_data(
        project=project,
        name=alert_name,
        entity=alert_entity,
        summary=alert_summary,
        event_kind=event_kind,
        criteria=alert_criteria,
    )

    services.api.crud.Alerts().store_alert(
        session=db,
        project=project,
        name=alert_name,
        alert_data=alert_data,
    )

    # trigger 3 events to activate the alert
    for _ in range(3):
        await fastapi.concurrency.run_in_threadpool(
            services.api.crud.Events().process_event,
            session=db,
            event_data=event_data,
            event_name=event_data.kind,
            project=project,
        )

    activations = services.api.crud.AlertActivation().list_alerts_activations(
        session=db, project=project
    )
    expected_notifications_states = [
        mlrun.common.schemas.NotificationState(kind="slack", status=""),
    ]
    expected_entity_id = (
        f"{alert_data.entities.ids[0]}.{event_data.value_dict.get('uid')}"
    )
    expected_activation_time = event_data.timestamp

    assert len(activations) == 1

    _assert_alert_activation(
        activations[0],
        alert_name=alert_name,
        project=project,
        entity_kind=alert_entity.kind,
        entity_id=expected_entity_id,
        notifications_states=expected_notifications_states,
        activation_time=expected_activation_time,
        number_of_events=alert_criteria.count,
    )


def _assert_alert_activation(
    alert_activation,
    alert_name,
    project,
    entity_kind,
    entity_id,
    notifications_states,
    activation_time,
    number_of_events,
):
    assert alert_activation.name == alert_name
    assert alert_activation.project == project
    assert alert_activation.entity_id == entity_id
    assert alert_activation.entity_kind == entity_kind
    assert alert_activation.number_of_events == number_of_events
    assert alert_activation.activation_time == activation_time
    assert alert_activation.notifications == notifications_states
