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


import pytest

import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.utils.helpers


@pytest.mark.parametrize(
    "summary, project, alert_name, entity_id, expected_str",
    [
        (
            "Model {{project}}/{{entity}} is drifting.",
            "my-project",
            None,
            "123",
            "Model my-project/123 is drifting.",
        ),
        (
            "Alert {{name}}: Model {{project}}/{{entity}} is drifting.",
            "my-project123",
            "alert",
            "ddd",
            "Alert alert: Model my-project123/ddd is drifting.",
        ),
        (
            "Model is drifting.",
            None,
            None,
            None,
            "Model is drifting.",
        ),
    ],
)
def test_summary_formatter(summary, project, alert_name, entity_id, expected_str):
    notification = mlrun.common.schemas.Notification(
        kind="slack",
        name="slack_drift",
        secret_params={
            "webhook": "https://hooks.slack.com/services/",
        },
        condition="oops",
    )

    event_kind = alert_objects.EventKind.DATA_DRIFT_SUSPECTED
    entity_kind = alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT

    project = "my-project" if project is None else project
    alert_name = "my-alert" if alert_name is None else alert_name
    entity_id = "123" if entity_id is None else entity_id

    alert = mlrun.common.schemas.AlertConfig(
        project=project,
        name=alert_name,
        summary=summary,
        severity=alert_objects.AlertSeverity.MEDIUM,
        entities=alert_objects.EventEntities(
            kind=entity_kind,
            project=project,
            ids=[entity_id],
        ),
        trigger=alert_objects.AlertTrigger(events=[event_kind]),
        notifications=[alert_objects.AlertNotification(notification=notification)],
    )
    alert_data = mlrun.common.schemas.Event(
        kind=event_kind,
        entity=alert_objects.EventEntities(
            kind=entity_kind, project=project, ids=[entity_id]
        ),
    )
    result = mlrun.utils.helpers.format_alert_summary(alert, alert_data)
    assert result == expected_str
