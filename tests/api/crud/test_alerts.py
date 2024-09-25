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
import server.api.crud
import tests.api.conftest


@pytest.mark.asyncio
async def test_process_event_no_cache(
    db: sqlalchemy.orm.Session,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    project = "project-name"
    alert_name = "my-alert"
    entity = mlrun.common.schemas.alert.EventEntities(
        kind=mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
        project=project,
        ids=[123],
    )
    event_kind = mlrun.common.schemas.alert.EventKind.DATA_DRIFT_SUSPECTED

    alert = mlrun.common.schemas.alert.AlertConfig(
        project=project,
        name=alert_name,
        summary="testing 1 2 3",
        severity=mlrun.common.schemas.alert.AlertSeverity.MEDIUM,
        entities=entity,
        trigger=mlrun.common.schemas.alert.AlertTrigger(events=[event_kind]),
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.MANUAL,
        notifications=[
            {
                "notification": {
                    "kind": "slack",
                    "name": "slack_drift",
                    "message": "Ay ay ay!",
                    "severity": "warning",
                    "when": ["now"],
                    "condition": "failed",
                    "secret_params": {
                        "webhook": "https://hooks.slack.com/services/",
                    },
                },
            },
        ],
    )

    server.api.crud.Alerts().store_alert(
        db, project=project, name=alert_name, alert_data=alert
    )

    event = mlrun.common.schemas.alert.Event(
        kind=event_kind,
        entity=entity,
    )

    await fastapi.concurrency.run_in_threadpool(
        server.api.crud.Alerts().process_event_no_cache, db, event.kind, event
    )

    alert = server.api.crud.Alerts().get_enriched_alert(
        db, project=project, name=alert_name
    )
    assert alert.state == mlrun.common.schemas.alert.AlertActiveState.ACTIVE


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
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    alert_name: str,
    expectation: AbstractContextManager,
):
    project = "project-name"
    entity = mlrun.common.schemas.alert.EventEntities(
        kind=mlrun.common.schemas.alert.EventEntityKind.JOB,
        project=project,
        ids=[123],
    )
    event_kind = mlrun.common.schemas.alert.EventKind.FAILED
    notification = mlrun.common.schemas.Notification(
        kind="slack",
        name="slack_notification",
        secret_params={
            "webhook": "https://hooks.slack.com/services/",
        },
        condition="oops",
    )

    alert_data = mlrun.common.schemas.alert.AlertConfig(
        project=project,
        name=alert_name,
        summary="The job has failed",
        severity=mlrun.common.schemas.alert.AlertSeverity.MEDIUM,
        entities=entity,
        trigger=mlrun.common.schemas.alert.AlertTrigger(events=[event_kind]),
        notifications=[
            mlrun.common.schemas.AlertNotification(notification=notification)
        ],
    )
    with expectation:
        server.api.crud.Alerts().store_alert(
            db, project=project, name=alert_name, alert_data=alert_data
        )
