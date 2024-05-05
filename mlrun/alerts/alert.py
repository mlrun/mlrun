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

from typing import Union

import mlrun
import mlrun.common.schemas.alert as alert_constants
from mlrun.common.schemas.notification import Notification
from mlrun.model import ModelObj


class AlertConfig(ModelObj):
    _dict_fields = [
        "project",
        "name",
        "description",
        "summary",
        "severity",
        "criteria",
        "reset_policy",
    ]

    def __init__(
        self,
        project: str,
        name: str,
        template: Union[alert_constants.AlertTemplate, str] = None,
        description: str = None,
        summary: str = None,
        severity: alert_constants.AlertSeverity = None,
        trigger: alert_constants.AlertTrigger = None,
        criteria: alert_constants.AlertCriteria = None,
        reset_policy: alert_constants.ResetPolicy = None,
        notifications: list[Notification] = None,
        entity: alert_constants.EventEntity = None,
        id: int = None,
        state: alert_constants.AlertActiveState = None,
        created: str = None,
        count: int = None,
    ):
        self.project = project
        self.name = name
        self.description = description
        self.summary = summary
        self.severity = severity
        self.trigger = trigger
        self.criteria = criteria
        self.reset_policy = reset_policy
        self.notifications = notifications or []
        self.entity = entity
        self.id = id
        self.state = state
        self.created = created
        self.count = count

        if template:
            self._apply_template(template)

    def to_dict(self, fields: list = None, exclude: list = None, strip: bool = False):
        data = super().to_dict(self._dict_fields)

        data["entity"] = (
            self.entity.dict() if not isinstance(self.entity, dict) else self.entity
        )
        data["notifications"] = [
            notification.dict() if not isinstance(notification, dict) else notification
            for notification in self.notifications
        ]
        data["trigger"] = (
            self.trigger.dict() if not isinstance(self.trigger, dict) else self.trigger
        )
        return data

    def from_dict(self, struct=None, fields=None, deprecated_fields: dict = None):
        new_obj = super().from_dict(struct, self._dict_fields)

        entity_data = struct.get("entity")
        if entity_data:
            entity_obj = alert_constants.EventEntity.parse_obj(entity_data)
            new_obj.entity = entity_obj

        notifications_data = struct.get("notifications")
        if notifications_data:
            notifications_objs = [
                Notification.parse_obj(notification_data)
                for notification_data in notifications_data
            ]
            new_obj.notifications = notifications_objs

        trigger_data = struct.get("trigger")
        if trigger_data:
            trigger_obj = alert_constants.AlertTrigger.parse_obj(trigger_data)
            new_obj.trigger = trigger_obj

        return new_obj

    def with_notifications(self, notifications: list[Notification]):
        if not isinstance(notifications, list) or not all(
            isinstance(item, Notification) for item in notifications
        ):
            raise ValueError("Notifications parameter must be a list of notifications")
        for notification in notifications:
            self.notifications.append(notification)
        return self

    def with_entity(self, entity: alert_constants.EventEntity):
        if not isinstance(entity, alert_constants.EventEntity):
            raise ValueError("entity parameter must be of type: EventEntity")
        self.entity = entity
        return self

    def _apply_template(self, template):
        if isinstance(template, str):
            db = mlrun.get_run_db()
            template = db.get_alert_template(template)

        # Extract parameters from the template and apply them to the AlertConfig object
        self.description = template.description
        self.severity = template.severity
        self.criteria = template.criteria
        self.trigger = template.trigger
        self.reset_policy = template.reset_policy
