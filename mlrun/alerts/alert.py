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
from datetime import datetime
from typing import Optional, Union

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
        "count",
        "created",
        "updated",
    ]
    _fields_to_serialize = ModelObj._fields_to_serialize + [
        "entities",
        "notifications",
        "trigger",
        "criteria",
    ]

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        template: Union[alert_objects.AlertTemplate, str] = None,
        description: Optional[str] = None,
        summary: Optional[str] = None,
        severity: alert_objects.AlertSeverity = None,
        trigger: alert_objects.AlertTrigger = None,
        criteria: alert_objects.AlertCriteria = None,
        reset_policy: alert_objects.ResetPolicy = None,
        notifications: Optional[list[alert_objects.AlertNotification]] = None,
        entities: alert_objects.EventEntities = None,
        id: Optional[int] = None,
        state: alert_objects.AlertActiveState = None,
        created: Optional[str] = None,
        count: Optional[int] = None,
        updated: Optional[str] = None,
        **kwargs,
    ):
        """Alert config object

        Example::

            # create an alert on endpoint_id, which will be triggered to slack if there is a "data_drift_detected" event
            # 3 times in the next hour.

            from mlrun.alerts import AlertConfig
            import mlrun.common.schemas.alert as alert_objects

            entity_kind = alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT
            entity_id = get_default_result_instance_fqn(endpoint_id)
            event_name = alert_objects.EventKind.DATA_DRIFT_DETECTED
            notification = mlrun.model.Notification(
                kind="slack",
                name="slack_notification",
                message="drift was detected",
                severity="warning",
                when=["now"],
                condition="failed",
                secret_params={
                    "webhook": "https://hooks.slack.com/",
                },
            ).to_dict()

            alert_data = AlertConfig(
                project="my-project",
                name="drift-alert",
                summary="a drift was detected",
                severity=alert_objects.AlertSeverity.LOW,
                entities=alert_objects.EventEntities(
                    kind=entity_kind, project="my-project", ids=[entity_id]
                ),
                trigger=alert_objects.AlertTrigger(events=[event_name]),
                criteria=alert_objects.AlertCriteria(count=3, period="1h"),
                notifications=[alert_objects.AlertNotification(notification=notification)],
            )
            project.store_alert_config(alert_data)

        :param project:        Name of the project to associate the alert with
        :param name:           Name of the alert
        :param template:       Optional parameter that allows creating an alert based on a predefined template.
                               You can pass either an AlertTemplate object or a string (the template name).
                               If a template is used, many fields of the alert will be auto-generated based on the
                               template.However, you still need to provide the following fields:
                               `name`, `project`, `entity`, `notifications`
        :param description:    Description of the alert
        :param summary:        Summary of the alert, will be sent in the generated notifications
        :param severity:       Severity of the alert
        :param trigger:        The events that will trigger this alert, may be a simple trigger based on events or
                               complex trigger which is based on a prometheus alert
        :param criteria:       When the alert will be triggered based on the specified number of events within the
                               defined time period.
        :param reset_policy:   When to clear the alert. Either "manual" for manual reset of the alert, or
                               "auto" if the criteria contains a time period
        :param notifications:  List of notifications to invoke once the alert is triggered
        :param entities:       Entities that the event relates to. The entity object will contain fields that
                               uniquely identify a given entity in the system
        :param id:             Internal id of the alert (user should not supply it)
        :param state:          State of the alert, may be active/inactive (user should not supply it)
        :param created:        When the alert is created (user should not supply it)
        :param count:          Internal counter of the alert (user should not supply it)
        :param updated:        The last update time of the alert (user should not supply it)
        """
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
        self._created = created
        self.count = count
        self._updated = updated

        if template:
            self._apply_template(template)

    @property
    def created(self) -> datetime:
        """
        Get the `created` field as a datetime object.
        """
        if isinstance(self._created, str):
            return datetime.fromisoformat(self._created)
        return self._created

    @created.setter
    def created(self, created):
        self._created = created

    @property
    def updated(self) -> datetime:
        """
        Get the `updated` field as a datetime object.
        """
        if isinstance(self._updated, str):
            return datetime.fromisoformat(self._updated)
        return self._updated

    @updated.setter
    def updated(self, updated):
        self._updated = updated

    def validate_required_fields(self):
        if not self.name:
            raise mlrun.errors.MLRunInvalidArgumentError("Alert name must be provided")

    def _serialize_field(
        self, struct: dict, field_name: Optional[str] = None, strip: bool = False
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

    def to_dict(
        self,
        fields: Optional[list] = None,
        exclude: Optional[list] = None,
        strip: bool = False,
    ):
        if self.entities is None:
            raise mlrun.errors.MLRunBadRequestError("Alert entity field is missing")
        if not self.notifications:
            raise mlrun.errors.MLRunBadRequestError(
                "Alert must have at least one notification"
            )
        return super().to_dict(self._dict_fields)

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: Optional[dict] = None
    ):
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

        # Apply parameters from the template to the AlertConfig object only if they are not already specified by the
        # user in the current configuration.
        # User-provided parameters will take precedence over corresponding template values
        self.summary = self.summary or template.summary
        self.severity = self.severity or template.severity
        self.criteria = self.criteria or template.criteria
        self.trigger = self.trigger or template.trigger
        self.reset_policy = self.reset_policy or template.reset_policy

    def list_activations(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        from_last_update: bool = False,
    ) -> list[mlrun.common.schemas.alert.AlertActivation]:
        """
        Retrieve a list of all alert activations.

        :param since: Filters for alert activations occurring after this timestamp.
        :param until: Filters for alert activations occurring before this timestamp.
        :param from_last_update: If set to True, retrieves alert activations since the alert's last update time.
                                 if both since and from_last_update=True are provided, from_last_update takes precedence
                                 and the since value will be overridden by the alert's last update timestamp.

        :returns: A list of alert activations matching the provided filters.
        """
        db = mlrun.get_run_db()
        if from_last_update and self._updated:
            since = self.updated

        return db.list_alert_activations(
            project=self.project,
            name=self.name,
            since=since,
            until=until,
        )

    def paginated_list_activations(
        self,
        *args,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        from_last_update: bool = False,
        **kwargs,
    ) -> tuple[mlrun.common.schemas.alert.AlertActivation, Optional[str]]:
        """
        List alerts activations with support for pagination and various filtering options.

        This method retrieves a paginated list of alert activations based on the specified filter parameters.
        Pagination is controlled using the `page`, `page_size`, and `page_token` parameters. The method
        will return a list of alert activations that match the filtering criteria provided.

        For detailed information about the parameters, refer to the list_activations method:
            See :py:func:`~list_activations` for more details.

        Examples::

            # Fetch first page of alert activations with page size of 5
            alert_activations, token = alert_config.paginated_list_activations(page_size=5)
            # Fetch next page using the pagination token from the previous response
            alert_activations, token = alert_config.paginated_list_activations(
                page_token=token
            )
            # Fetch alert activations for a specific page (e.g., page 3)
            alert_activations, token = alert_config.paginated_list_activations(
                page=3, page_size=5
            )

            # Automatically iterate over all pages without explicitly specifying the page number
            alert_activations = []
            token = None
            while True:
                page_alert_activations, token = alert_config.paginated_list_activations(
                    page_token=token, page_size=5
                )
                alert_activations.extend(page_alert_activations)

                # If token is None and page_alert_activations is empty, we've reached the end (no more activations).
                # If token is None and page_alert_activations is not empty, we've fetched the last page of activations.
                if not token:
                    break
            print(f"Total alert activations retrieved: {len(alert_activations)}")

        :param page: The page number to retrieve. If not provided, the next page will be retrieved.
        :param page_size: The number of items per page to retrieve. Up to `page_size` responses are expected.
        :param page_token: A pagination token used to retrieve the next page of results. Should not be provided
            for the first request.
        :param from_last_update: If set to True, retrieves alert activations since the alert's last update time.

        :returns: A tuple containing the list of alert activations and an optional `page_token` for pagination.
        """
        if from_last_update and self._updated:
            kwargs["since"] = self.updated

        db = mlrun.get_run_db()
        return db.paginated_list_alert_activations(
            *args,
            project=self.project,
            name=self.name,
            page=page,
            page_size=page_size,
            page_token=page_token,
            **kwargs,
        )
