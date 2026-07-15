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

import datetime
import typing

import pydantic.v1

from .._shared.notification import (
    NotificationKind,
    NotificationSeverity,
    NotificationStatus,
)


class Notification(pydantic.v1.BaseModel):
    """
    Notification object schema

    :param kind: notification implementation kind - slack, webhook, etc.
    :param name: for logging and identification
    :param message: message content in the notification
    :param severity: severity to display in the notification
    :param when: list of statuses to trigger the notification: 'running', 'completed', 'error'
    :param condition: optional condition to trigger the notification, a jinja2 expression that can use run data
                      to evaluate if the notification should be sent in addition to the 'when' statuses.
                      e.g.: '{{ run["status"]["results"]["accuracy"] < 0.9}}'
    :param params: Implementation specific parameters for the notification implementation (e.g. slack webhook url,
                   git repository details, etc.)
    :param secret_params: secret parameters for the notification implementation, same as params but will be stored
                          in a k8s secret and passed as a secret reference to the implementation.
    :param status: notification status - pending, sent, error
    :param sent_time: time the notification was sent
    :param reason: failure reason if the notification failed to send
    """

    kind: NotificationKind
    name: str
    message: str | None = None
    severity: NotificationSeverity | None = None
    when: list[str] | None = None
    condition: str | None = None
    params: dict[str, typing.Any] | None = None
    status: NotificationStatus | None = None
    sent_time: typing.Union[str, datetime.datetime] | None = None
    secret_params: dict[str, typing.Any] | None = None
    reason: str | None = None


class SetNotificationRequest(pydantic.v1.BaseModel):
    notifications: list[Notification] = None


class NotificationSummary(pydantic.v1.BaseModel):
    failed: int = 0
    succeeded: int = 0


class NotificationState(pydantic.v1.BaseModel):
    kind: str
    err: str | None  # empty error means that the notifications were sent successfully
    summary: NotificationSummary
