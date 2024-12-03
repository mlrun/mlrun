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
import collections
import datetime
from typing import Optional, Union

import sqlalchemy.orm

import mlrun.common.schemas.alert
import mlrun.utils.singleton

import framework.utils.singletons.db


class AlertActivation(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_alert_activation(
        self,
        session: sqlalchemy.orm.Session,
        alert_data: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
    ) -> Optional[str]:
        notifications_states = self._prepare_notification_states(
            alert_data.notifications
        )
        return framework.utils.singletons.db.get_db().store_alert_activation(
            session, alert_data, event_data, notifications_states
        )

    @staticmethod
    def _prepare_notification_states(
        notifications: list[mlrun.common.schemas.AlertNotification],
    ) -> list[mlrun.common.schemas.NotificationState]:
        """
        Processes a list of alert notifications to construct a list of NotificationState objects.

        Each NotificationState represents a unique type of notification (e.g., "slack", "email") and its status.
        For each notification type, this method aggregates error messages if any notifications of that type have failed.
        The resulting NotificationState has:
        - An empty 'err' if all notifications of that type succeeded.
        - An 'err' with all unique errors if all notifications of that type failed.
        - An 'err' with unique errors if some, but not all, notifications of that type failed.
        """

        notification_errors = collections.defaultdict(
            lambda: {
                "errors": set(),
                "success_count": 0,
                "failed_count": 0,
            },
        )

        # process each notification and gather errors by type
        for alert_notification in notifications:
            kind = alert_notification.notification.kind
            reason = alert_notification.notification.reason

            # count successes, failures and collect unique errors for failures
            if reason:
                notification_errors[kind]["errors"].add(reason)
                notification_errors[kind]["failed_count"] += 1
            else:
                notification_errors[kind]["success_count"] += 1

        # construct NotificationState objects based on the aggregated error data
        notification_states = []
        for kind, status_info in notification_errors.items():
            errors = list(status_info["errors"])
            success_count = status_info.get("success_count", 0)
            failed_count = status_info.get("failed_count", 0)

            if errors:
                if success_count == 0:
                    error_message = (
                        f"All {kind} notifications failed. Errors: {', '.join(errors)}"
                    )
                else:
                    error_message = (
                        f"Some {kind} notifications failed. Errors: {', '.join(errors)}"
                    )
            else:
                # indicates success if there are no errors
                error_message = ""

            notification_states.append(
                mlrun.common.schemas.NotificationState(
                    kind=kind,
                    err=error_message,
                    summary=mlrun.common.schemas.NotificationSummary(
                        failed=failed_count,
                        succeeded=success_count,
                    ),
                )
            )

        return notification_states

    def list_alert_activations(
        self,
        session: sqlalchemy.orm.Session,
        projects_with_creation_time: list[tuple[str, datetime.datetime]],
        name: Optional[str] = None,
        since: Optional[datetime.datetime] = None,
        until: Optional[datetime.datetime] = None,
        entity: Optional[str] = None,
        severity: Optional[
            list[Union[mlrun.common.schemas.alert.AlertSeverity, str]]
        ] = None,
        entity_kind: Optional[
            Union[mlrun.common.schemas.alert.EventEntityKind, str]
        ] = None,
        event_kind: Optional[Union[mlrun.common.schemas.alert.EventKind, str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> list[mlrun.common.schemas.AlertActivation]:
        return framework.utils.singletons.db.get_db().list_alert_activations(
            session=session,
            projects_with_creation_time=projects_with_creation_time,
            name=name,
            since=since,
            until=until,
            entity=entity,
            severity=severity,
            entity_kind=entity_kind,
            event_kind=event_kind,
            page=page,
            page_size=page_size,
        )
