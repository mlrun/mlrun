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
from datetime import datetime
from typing import Annotated, Optional, Union

import pydantic

from mlrun.common.schemas.notification import Notification
from mlrun.common.types import StrEnum


class EventEntityKind(StrEnum):
    MODEL_ENDPOINT_RESULT = "model-endpoint-result"
    MODEL_MONITORING_APPLICATION = "model-monitoring-application"
    JOB = "job"


class EventEntities(pydantic.BaseModel):
    kind: EventEntityKind
    project: str
    ids: pydantic.conlist(str, min_items=1, max_items=1)


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


class Event(pydantic.BaseModel):
    kind: EventKind
    timestamp: Union[str, datetime] = None  # occurrence time
    entity: EventEntities
    value_dict: Optional[dict] = pydantic.Field(default_factory=dict)

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
class AlertTrigger(pydantic.BaseModel):
    events: list[EventKind] = []
    prometheus_alert: str = None

    def __eq__(self, other):
        return (
            self.prometheus_alert == other.prometheus_alert
            and self.events == other.events
        )


class AlertCriteria(pydantic.BaseModel):
    count: Annotated[
        int,
        pydantic.Field(
            description="Number of events to wait until notification is sent"
        ),
    ] = 1
    period: Annotated[
        str,
        pydantic.Field(
            description="Time period during which event occurred. e.g. 1d, 3h, 5m, 15s"
        ),
    ] = None

    def __eq__(self, other):
        return self.count == other.count and self.period == other.period


class ResetPolicy(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


class AlertNotification(pydantic.BaseModel):
    notification: Notification
    cooldown_period: Annotated[
        str,
        pydantic.Field(
            description="Period during which notifications "
            "will not be sent after initial send. The format of this would be in time."
            " e.g. 1d, 3h, 5m, 15s"
        ),
    ] = None


class AlertConfig(pydantic.BaseModel):
    project: str
    id: int = None
    name: str
    description: Optional[str] = ""
    summary: Annotated[
        str,
        pydantic.Field(
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
    notifications: pydantic.conlist(AlertNotification, min_items=1)
    state: AlertActiveState = AlertActiveState.INACTIVE
    count: Optional[int] = 0

    def get_raw_notifications(self) -> list[Notification]:
        return [
            alert_notification.notification for alert_notification in self.notifications
        ]


class AlertsModes(StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class AlertTemplate(
    pydantic.BaseModel
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
