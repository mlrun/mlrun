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

import mlrun.common.schemas.alert
import mlrun.common.schemas.alert as alert_objects


def generate_alert_data(
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
