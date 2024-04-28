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
    MODEL = "model"
    JOB = "job"


class EventEntity(pydantic.BaseModel):
    kind: EventEntityKind
    project: str
    id: str


class EventKind(StrEnum):
    DRIFT_DETECTED = "drift_detected"
    DRIFT_SUSPECTED = "drift_suspected"
    FAILED = "failed"


_event_kind_entity_map = {
    EventKind.DRIFT_SUSPECTED: [EventEntityKind.MODEL],
    EventKind.DRIFT_DETECTED: [EventEntityKind.MODEL],
    EventKind.FAILED: [EventEntityKind.JOB],
}


class Event(pydantic.BaseModel):
    kind: EventKind
    timestamp: Union[str, datetime] = None  # occurrence time
    entity: EventEntity
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


class AlertCriteria(pydantic.BaseModel):
    count: Annotated[
        int,
        pydantic.Field(
            description="Number of events to wait until notification is sent"
        ),
    ] = 0
    period: Annotated[
        str,
        pydantic.Field(
            description="Time period during which event occurred. e.g. 1d, 3h, 5m, 15s"
        ),
    ] = None


class ResetPolicy(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


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
                "e.g. 'Model {{ $project }}/{{ $entity }} is drifting.'"
            )
        ),
    ]
    created: Union[str, datetime] = None
    severity: AlertSeverity
    entity: EventEntity
    trigger: AlertTrigger
    criteria: Optional[AlertCriteria]
    reset_policy: ResetPolicy = ResetPolicy.MANUAL
    notifications: pydantic.conlist(Notification, min_items=1)
    state: AlertActiveState = AlertActiveState.INACTIVE
    count: Optional[int] = 0


class AlertsModes(StrEnum):
    enabled = "enabled"
    disabled = "disabled"
