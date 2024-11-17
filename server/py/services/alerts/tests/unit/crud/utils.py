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

from typing import Optional

import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.notification as notification_objects


def generate_alert_data(
    project: str,
    name: str,
    entity: alert_objects.EventEntities,
    summary: str,
    event_kind: alert_objects.EventKind,
    description: Optional[str] = None,
    severity: alert_objects.AlertSeverity = alert_objects.AlertSeverity.LOW,
    notifications: Optional[list[notification_objects.Notification]] = None,
    criteria: alert_objects.AlertCriteria = None,
    reset_policy: alert_objects.ResetPolicy = alert_objects.ResetPolicy.AUTO,
):
    trigger = alert_objects.AlertTrigger(events=[event_kind])
    if notifications is None:
        notification = notification_objects.Notification(
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
