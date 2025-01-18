# Copyright 2023 Iguazio
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
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Any, Callable, Optional, Union

import pydantic.v1

import mlrun.common.schemas.notification as notification_objects
from mlrun.common.types import StrEnum


class EventEntityKind(StrEnum):
    MODEL_ENDPOINT_RESULT = "model-endpoint-result"
    MODEL_MONITORING_APPLICATION = "model-monitoring-application"
    JOB = "job"


class EventEntities(pydantic.v1.BaseModel):
    kind: EventEntityKind
    project: str
    ids: pydantic.v1.conlist(str, min_items=1, max_items=1)


class EventKind(StrEnum):
    DATA_DRIFT_DETECTED = "data-drift-detected"
    DATA_DRIFT_SUSPECTED = "data-drift-suspected"
    CONCEPT_DRIFT_DETECTED = "concept-drift-detected"
    CONCEPT_DRIFT_SUSPECTED = "concept-drift-suspected"
    MODEL_PERFORMANCE_DETECTED = "model-performance-detected"
    MODEL_PERFORMANCE_SUSPECTED = "model-performance-suspected"
    SYSTEM_PERFORMANCE_DETECTED = "system-performance-detected"
    SYSTEM_PERFORMANCE_SUSPECTED = "system-performance-suspected"
    MM_APP_ANOMALY_DETECTED = "mm-app-anomaly-detected"
    MM_APP_ANOMALY_SUSPECTED = "mm-app-anomaly-suspected"
    MM_APP_FAILED = "mm-app-failed"
    FAILED = "failed"


_event_kind_entity_map = {
    EventKind.DATA_DRIFT_SUSPECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.DATA_DRIFT_DETECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.CONCEPT_DRIFT_DETECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.CONCEPT_DRIFT_SUSPECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.MODEL_PERFORMANCE_DETECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.MODEL_PERFORMANCE_SUSPECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.SYSTEM_PERFORMANCE_DETECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.SYSTEM_PERFORMANCE_SUSPECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.MM_APP_ANOMALY_DETECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.MM_APP_ANOMALY_SUSPECTED: [EventEntityKind.MODEL_ENDPOINT_RESULT],
    EventKind.MM_APP_FAILED: [EventEntityKind.MODEL_MONITORING_APPLICATION],
    EventKind.FAILED: [EventEntityKind.JOB],
}


class Event(pydantic.v1.BaseModel):
    kind: EventKind
    timestamp: Union[str, datetime] = None  # occurrence time
    entity: EventEntities
    value_dict: Optional[dict] = pydantic.v1.Field(default_factory=dict)

    def is_valid(self):
        return self.entity.kind in _event_kind_entity_map[self.kind]


class AlertActiveState(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# what should trigger the alert. must be either event (at least 1), or prometheus query
class AlertTrigger(pydantic.v1.BaseModel):
    events: list[EventKind] = []
    prometheus_alert: str = None

    def __eq__(self, other):
        return (
            self.prometheus_alert == other.prometheus_alert
            and self.events == other.events
        )


class AlertCriteria(pydantic.v1.BaseModel):
    count: Annotated[
        int,
        pydantic.v1.Field(
            description="Number of events to wait until notification is sent"
        ),
    ] = 1
    period: Annotated[
        str,
        pydantic.v1.Field(
            description="Time period during which event occurred. e.g. 1d, 3h, 5m, 15s"
        ),
    ] = None

    def __eq__(self, other):
        return self.count == other.count and self.period == other.period


class ResetPolicy(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


class AlertNotification(pydantic.v1.BaseModel):
    notification: notification_objects.Notification
    cooldown_period: Annotated[
        str,
        pydantic.v1.Field(
            description="Period during which notifications "
            "will not be sent after initial send. The format of this would be in time."
            " e.g. 1d, 3h, 5m, 15s"
        ),
    ] = None


class AlertConfig(pydantic.v1.BaseModel):
    project: str
    id: int = None
    name: str
    description: Optional[str] = ""
    summary: Annotated[
        str,
        pydantic.v1.Field(
            description=(
                "String to be sent in the notifications generated."
                "e.g. 'Model {{project}}/{{entity}} is drifting.'"
                "Supported variables: project, entity, name"
            )
        ),
    ]
    created: Union[str, datetime] = None
    severity: AlertSeverity
    entities: EventEntities
    trigger: AlertTrigger
    criteria: Optional[AlertCriteria]
    reset_policy: ResetPolicy = ResetPolicy.AUTO
    notifications: pydantic.v1.conlist(AlertNotification, min_items=1)
    state: AlertActiveState = AlertActiveState.INACTIVE
    count: Optional[int] = 0
    updated: datetime = None

    class Config:
        extra = pydantic.v1.Extra.allow

    def get_raw_notifications(self) -> list[notification_objects.Notification]:
        return [
            alert_notification.notification for alert_notification in self.notifications
        ]


class AlertsModes(StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class AlertTemplate(
    pydantic.v1.BaseModel
):  # Template fields that are not shared with created configs
    template_id: int = None
    template_name: str
    template_description: Optional[str] = (
        "String explaining the purpose of this template"
    )

    # A property that identifies templates that were created by the system and cannot be modified/deleted by the user
    system_generated: bool = False

    # AlertConfig fields that are pre-defined
    summary: Optional[str] = (
        "String to be sent in the generated notifications e.g. 'Model {{project}}/{{entity}} is drifting.'"
        "See AlertConfig.summary description"
    )
    severity: AlertSeverity
    trigger: AlertTrigger
    criteria: Optional[AlertCriteria]
    reset_policy: ResetPolicy = ResetPolicy.AUTO

    # This is slightly different than __eq__ as it doesn't compare everything
    def templates_differ(self, other):
        return (
            self.template_description != other.template_description
            or self.summary != other.summary
            or self.severity != other.severity
            or self.trigger != other.trigger
            or self.reset_policy != other.reset_policy
            or self.criteria != other.criteria
        )


class AlertActivation(pydantic.v1.BaseModel):
    id: int
    name: str
    project: str
    severity: AlertSeverity
    activation_time: datetime
    entity_id: str
    entity_kind: EventEntityKind
    criteria: AlertCriteria
    event_kind: EventKind
    number_of_events: int
    notifications: list[notification_objects.NotificationState]
    reset_time: Optional[datetime] = None

    def group_key(self, attributes: list[str]) -> Union[Any, tuple]:
        """
        Dynamically create a key for grouping based on the provided attributes.
        - If there's only one attribute, return the value directly (not a single-element tuple).
        - If there are multiple attributes, return them as a tuple for grouping.

        This ensures grouping behaves intuitively without redundant tuple representations.
        """
        if len(attributes) == 1:
            # Avoid single-element tuple like (high,) when only one grouping attribute is used
            return getattr(self, attributes[0])
        # Otherwise, return a tuple of all specified attributes
        return tuple(getattr(self, attr) for attr in attributes)


class AlertActivations(pydantic.v1.BaseModel):
    activations: list[AlertActivation]
    pagination: Optional[dict]

    def __iter__(self) -> Iterator[AlertActivation]:
        return iter(self.activations)

    def __getitem__(self, index: int) -> AlertActivation:
        return self.activations[index]

    def __len__(self) -> int:
        return len(self.activations)

    def group_by(self, *attributes: str) -> dict:
        """
        Group alert activations by specified attributes.

        Args:
        :param attributes: Attributes to group by.

        :returns: A dictionary where keys are tuples of attribute values and values are lists of
            AlertActivation objects.

        Example:
            # Group by project and severity
            grouped = activations.group_by("project", "severity")
        """
        grouped = defaultdict(list)
        for activation in self.activations:
            key = activation.group_key(attributes)
            grouped[key].append(activation)
        return dict(grouped)

    def aggregate_by(
        self,
        group_by_attrs: list[str],
        aggregation_function: Callable[[list[AlertActivation]], Any],
    ) -> dict:
        """
        Aggregate alert activations by specified attributes using a given aggregation function.

        Args:
        :param group_by_attrs: Attributes to group by.
        :param aggregation_function: Function to aggregate grouped activations.

        :returns: A dictionary where keys are tuples of attribute values and values are the result
            of the aggregation function.

        Example:
            # Aggregate by name and entity_id and count number of activations in each group
            activations.aggregate_by(["name", "entity_id"], lambda activations: len(activations))
        """
        grouped = self.group_by(*group_by_attrs)
        aggregated = {
            key: aggregation_function(activations)
            for key, activations in grouped.items()
        }
        return aggregated
