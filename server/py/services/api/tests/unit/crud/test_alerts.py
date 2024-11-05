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

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import fastapi.concurrency
import pytest
import sqlalchemy.orm

import mlrun.common.schemas.alert
import mlrun.common.schemas.alert as alert_objects
import services.api.crud
import services.api.tests.unit.conftest


@pytest.fixture
def reset_alert_caches():
    yield
    services.api.crud.Alerts()._alert_cache.cache_clear()
    services.api.crud.Alerts()._alert_state_cache.cache_clear()


@pytest.mark.asyncio
async def test_process_event_no_cache(
    db: sqlalchemy.orm.Session,
    k8s_secrets_mock: services.api.tests.unit.conftest.K8sSecretsMock,
):
    project = "project-name"
    alert_name = "my-alert"
    alert_summary = "testing 1 2 3"
    alert_reset_policy = alert_objects.ResetPolicy.MANUAL
    alert_entity = alert_objects.EventEntities(
        kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
        project=project,
        ids=[123],
    )
    event_kind = alert_objects.EventKind.DATA_DRIFT_SUSPECTED
    alert_trigger = alert_objects.AlertTrigger(events=[event_kind])

    alert_data = _generate_alert_data(
        project=project,
        name=alert_name,
        entity=alert_entity,
        summary=alert_summary,
        trigger=alert_trigger,
        reset_policy=alert_reset_policy,
    )

    services.api.crud.Alerts().store_alert(
        session=db,
        project=project,
        name=alert_name,
        alert_data=alert_data,
    )

    event = alert_objects.Event(kind=event_kind, entity=alert_entity)

    await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Alerts().process_event_no_cache, db, event.kind, event
    )

    alert = services.api.crud.Alerts().get_enriched_alert(
        session=db,
        project=project,
        name=alert_name,
    )
    assert alert.state == alert_objects.AlertActiveState.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alert_name, expectation",
    [
        ("invalid_name?", pytest.raises(mlrun.errors.MLRunBadRequestError)),
        ("name with spaces", pytest.raises(mlrun.errors.MLRunBadRequestError)),
        ("invalid/name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
        ("invalid@name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
        ("invalid_name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
        ("$indalid_name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
        ("valid-name", does_not_raise()),
        ("valid-name-123", does_not_raise()),
    ],
)
async def test_validate_alert_name(
    db: sqlalchemy.orm.Session,
    k8s_secrets_mock: services.api.tests.unit.conftest.K8sSecretsMock,
    alert_name: str,
    expectation: AbstractContextManager,
):
    project = "project-name"
    alert_summary = "The job has failed"
    alert_entity = alert_objects.EventEntities(
        kind=alert_objects.EventEntityKind.JOB,
        project=project,
        ids=[123],
    )
    event_kind = alert_objects.EventKind.FAILED
    alert_trigger = alert_objects.AlertTrigger(events=[event_kind])

    alert_data = _generate_alert_data(
        project=project,
        name=alert_name,
        entity=alert_entity,
        summary=alert_summary,
        trigger=alert_trigger,
    )
    with expectation:
        services.api.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )


@pytest.mark.parametrize(
    "modify_field, modified_value, should_reset",
    [
        # Non-functional fields:
        ("summary", "The job has failed again", False),
        ("description", "Job failure detected", False),
        ("severity", alert_objects.AlertSeverity.HIGH, False),
        (
            "notifications",
            [
                alert_objects.AlertNotification(
                    notification=mlrun.common.schemas.Notification(
                        kind="webhook",
                        name="webhook_notification",
                        params={
                            "url": "some-webhook-url",
                        },
                    )
                )
            ],
            False,
        ),
        ("reset_policy", alert_objects.ResetPolicy.AUTO, False),
        # Functional fields:
        (
            "entities",
            alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.JOB,
                project="project-name",
                ids=[456],
            ),
            True,
        ),
        (
            "entities",
            alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
                project="project-name",
                ids=[123],
            ),
            True,
        ),
        (
            "trigger",
            alert_objects.AlertTrigger(
                events=[alert_objects.EventKind.DATA_DRIFT_DETECTED]
            ),
            True,
        ),
        (
            "criteria",
            alert_objects.AlertCriteria(
                count=5,
                period="10m",
            ),
            True,
        ),
        # Test multiple modifications
        (
            ["summary", "severity"],
            [
                "Job has failed again",
                alert_objects.AlertSeverity.HIGH,
            ],
            False,
        ),
        (
            ["summary", "severity", "reset_policy"],
            [
                "Job has failed again",
                alert_objects.AlertSeverity.HIGH,
                alert_objects.ResetPolicy.AUTO,
            ],
            False,
        ),
        (
            ["summary", "severity", "trigger"],
            [
                "Job has failed again",
                alert_objects.AlertSeverity.HIGH,
                alert_objects.AlertTrigger(
                    events=[alert_objects.EventKind.DATA_DRIFT_SUSPECTED]
                ),
            ],
            True,
        ),
        (
            ["criteria", "trigger"],
            [
                alert_objects.AlertCriteria(
                    count=3,
                    period="10m",
                ),
                alert_objects.AlertTrigger(
                    events=[alert_objects.EventKind.DATA_DRIFT_SUSPECTED]
                ),
            ],
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "force_reset",
    [False, True],
)
async def test_alert_reset_with_fields_updates(
    db: sqlalchemy.orm.Session,
    modify_field,
    modified_value,
    should_reset,
    force_reset,
    k8s_secrets_mock: services.api.tests.unit.conftest.K8sSecretsMock,
    reset_alert_caches,
):
    project = "project-name"
    alert_name = "failed-alert"
    alert_summary = "The job has failed"
    alert_reset_policy = alert_objects.ResetPolicy.MANUAL
    alert_entity = alert_objects.EventEntities(
        kind=alert_objects.EventEntityKind.JOB,
        project=project,
        ids=[123],
    )
    event_kind = alert_objects.EventKind.FAILED
    alert_trigger = alert_objects.AlertTrigger(events=[event_kind])

    alert_data = _generate_alert_data(
        project=project,
        name=alert_name,
        entity=alert_entity,
        summary=alert_summary,
        trigger=alert_trigger,
        reset_policy=alert_reset_policy,
    )

    # store the initial alert
    services.api.crud.Alerts().store_alert(
        session=db,
        project=project,
        name=alert_name,
        alert_data=alert_data,
    )

    # activate the alert
    event = alert_objects.Event(kind=event_kind, entity=alert_entity)
    await fastapi.concurrency.run_in_threadpool(
        services.api.crud.Alerts().process_event_no_cache,
        db,
        event.kind,
        event,
    )
    alert = services.api.crud.Alerts().get_enriched_alert(
        session=db,
        project=project,
        name=alert_name,
    )
    assert alert.state == alert_objects.AlertActiveState.ACTIVE

    # modify the alert data based on the parameterized field
    if isinstance(modify_field, list):
        for field, value in zip(modify_field, modified_value):
            setattr(alert_data, field, value)
    else:
        setattr(alert_data, modify_field, modified_value)

    # store the modified alert
    services.api.crud.Alerts().store_alert(
        session=db,
        project=project,
        name=alert_name,
        alert_data=alert_data,
        force_reset=force_reset,
    )

    # fetch the updated alert
    alert = services.api.crud.Alerts().get_enriched_alert(
        session=db,
        project=project,
        name=alert_name,
    )

    # validate the state based on whether it should have reset
    expected_state = (
        alert_objects.AlertActiveState.INACTIVE
        if should_reset or force_reset
        else alert_objects.AlertActiveState.ACTIVE
    )
    assert alert.state == expected_state


def _generate_alert_data(
    project,
    name,
    entity,
    summary,
    trigger,
    description=None,
    severity=alert_objects.AlertSeverity.LOW,
    notifications=None,
    criteria=None,
    reset_policy=alert_objects.ResetPolicy.AUTO,
):
    if notifications is None:
        notification = mlrun.common.schemas.Notification(
            kind="slack",
            name="slack_notification",
            secret_params={
                "webhook": "https://hooks.slack.com/services/",
            },
        )
        notifications = [alert_objects.AlertNotification(notification=notification)]
    return alert_objects.AlertConfig(
        project=project,
        name=name,
        description=description,
        summary=summary,
        severity=severity,
        entities=entity,
        trigger=trigger,
        criteria=criteria,
        notifications=notifications,
        reset_policy=reset_policy,
    )
