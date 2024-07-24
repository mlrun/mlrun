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
import mlrun.common.schemas.alert as alert_objects
from mlrun.model import ModelObj


class AlertConfig(ModelObj):
    _dict_fields = [
        "project",
        "name",
        "description",
        "summary",
        "severity",
        "reset_policy",
        "state",
    ]
    _fields_to_serialize = ModelObj._fields_to_serialize + [
        "entities",
        "notifications",
        "trigger",
        "criteria",
    ]

    def __init__(
        self,
        project: str = None,
        name: str = None,
        template: Union[alert_objects.AlertTemplate, str] = None,
        description: str = None,
        summary: str = None,
        severity: alert_objects.AlertSeverity = None,
        trigger: alert_objects.AlertTrigger = None,
        criteria: alert_objects.AlertCriteria = None,
        reset_policy: alert_objects.ResetPolicy = None,
        notifications: list[alert_objects.AlertNotification] = None,
        entities: alert_objects.EventEntities = None,
        id: int = None,
        state: alert_objects.AlertActiveState = None,
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
        self.entities = entities
        self.id = id
        self.state = state
        self.created = created
        self.count = count

        if template:
            self._apply_template(template)

    def validate_required_fields(self):
        if not self.project or not self.name:
            raise mlrun.errors.MLRunBadRequestError("Project and name must be provided")

    def _serialize_field(
        self, struct: dict, field_name: str = None, strip: bool = False
    ):
        if field_name == "entities":
            if self.entities:
                return (
                    self.entities.dict()
                    if not isinstance(self.entities, dict)
                    else self.entities
                )
            return None
        if field_name == "notifications":
            if self.notifications:
                return [
                    notification_data.dict()
                    if not isinstance(notification_data, dict)
                    else notification_data
                    for notification_data in self.notifications
                ]
            return None
        if field_name == "trigger":
            if self.trigger:
                return (
                    self.trigger.dict()
                    if not isinstance(self.trigger, dict)
                    else self.trigger
                )
            return None
        if field_name == "criteria":
            if self.criteria:
                return (
                    self.criteria.dict()
                    if not isinstance(self.criteria, dict)
                    else self.criteria
                )
            return None
        return super()._serialize_field(struct, field_name, strip)

    def to_dict(self, fields: list = None, exclude: list = None, strip: bool = False):
        if self.entities is None:
            raise mlrun.errors.MLRunBadRequestError("Alert entity field is missing")
        if not self.notifications:
            raise mlrun.errors.MLRunBadRequestError(
                "Alert must have at least one notification"
            )
        return super().to_dict(self._dict_fields)

    @classmethod
    def from_dict(cls, struct=None, fields=None, deprecated_fields: dict = None):
        new_obj = super().from_dict(struct, fields=fields)

        entity_data = struct.get("entities")
        if entity_data:
            entity_obj = alert_objects.EventEntities.parse_obj(entity_data)
            new_obj.entities = entity_obj

        notifications_data = struct.get("notifications")
        if notifications_data:
            notifications_objs = [
                alert_objects.AlertNotification.parse_obj(notification)
                for notification in notifications_data
            ]
            new_obj.notifications = notifications_objs

        trigger_data = struct.get("trigger")
        if trigger_data:
            trigger_obj = alert_objects.AlertTrigger.parse_obj(trigger_data)
            new_obj.trigger = trigger_obj

        criteria_data = struct.get("criteria")
        if criteria_data:
            criteria_obj = alert_objects.AlertCriteria.parse_obj(criteria_data)
            new_obj.criteria = criteria_obj
        return new_obj

    def with_notifications(self, notifications: list[alert_objects.AlertNotification]):
        if not isinstance(notifications, list) or not all(
            isinstance(item, alert_objects.AlertNotification) for item in notifications
        ):
            raise ValueError(
                "Notifications parameter must be a list of AlertNotification"
            )
        for notification_data in notifications:
            self.notifications.append(notification_data)
        return self

    def with_entities(self, entities: alert_objects.EventEntities):
        if not isinstance(entities, alert_objects.EventEntities):
            raise ValueError("Entities parameter must be of type: EventEntities")
        self.entities = entities
        return self

    def _apply_template(self, template):
        if isinstance(template, str):
            db = mlrun.get_run_db()
            template = db.get_alert_template(template)

        # Extract parameters from the template and apply them to the AlertConfig object
        self.summary = template.summary
        self.severity = template.severity
        self.criteria = template.criteria
        self.trigger = template.trigger
        self.reset_policy = template.reset_policy
