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
import enum
import typing

import pydantic

import mlrun.common.types


class NotificationKind(mlrun.common.types.StrEnum):
    """Currently, the supported notification kinds and their params are as follows:"""

    console: str = "console"
    """no params, local only"""

    git: str = "git"
    """
    **token** - The git token to use for the git notification.\n
    **repo** - The git repo to which to send the notification.\n
    **issue** - The git issue to which to send the notification.\n
    **merge_request** -
     In GitLab (as opposed to GitHub), merge requests and issues are separate entities.
     If using merge request, the issue will be ignored, and vice versa.\n
    **server** - The git server to which to send the notification.\n
    **gitlab** - (bool) Whether the git server is GitLab or not.\n
    """

    ipython: str = "ipython"
    """no params, local only"""

    slack: str = "slack"
    """**webhook** - The slack webhook to which to send the notification."""

    webhook: str = "webhook"
    """
    **url** - The webhook url to which to send the notification.\n
    **method** - The http method to use when sending the notification (GET, POST, PUT, etc…).\n
    **headers** - (dict) The http headers to send with the notification.\n
    **override_body** -
     (dict) The body to send with the notification. If not specified, the
     default body will be a dictionary containing `name`, `message`, `severity`, and a `runs` list of the
     completed runs. You can also add the run's details.\n
     Example::

                "override_body": {"message":"Run Completed {{ runs }}"
                # Results would look like:
                "message": "Run Completed [{'project': 'my-project', 'name': 'my-function', 'host': <run-host>,
                         'status': {'state': 'completed', 'results': <run-results>}}]"
    **verify_ssl** -
     (bool) Whether SSL certificates are validated during HTTP requests or not.
     The default is set to True.\n
    """


class NotificationSeverity(mlrun.common.types.StrEnum):
    INFO = "info"
    DEBUG = "debug"
    VERBOSE = "verbose"
    WARNING = "warning"
    ERROR = "error"


class NotificationStatus(mlrun.common.types.StrEnum):
    PENDING = "pending"
    SENT = "sent"
    ERROR = "error"


class NotificationLimits(enum.Enum):
    max_params_size = (
        900 * 1024
    )  # 900KB (k8s secret size limit is 1MB minus buffer for metadata)


class Notification(pydantic.BaseModel):
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
    message: typing.Optional[str] = None
    severity: typing.Optional[NotificationSeverity] = None
    when: typing.Optional[list[str]] = None
    condition: typing.Optional[str] = None
    params: typing.Optional[dict[str, typing.Any]] = None
    status: typing.Optional[NotificationStatus] = None
    sent_time: typing.Optional[typing.Union[str, datetime.datetime]] = None
    secret_params: typing.Optional[dict[str, typing.Any]] = None
    reason: typing.Optional[str] = None


class SetNotificationRequest(pydantic.BaseModel):
    notifications: list[Notification] = None
