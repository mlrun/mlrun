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

import sqlalchemy.orm

import mlrun.common.schemas.alert
import mlrun.utils.singleton

import services.api.utils.singletons.db


class AlertActivation(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_alert_activation(
        self,
        session: sqlalchemy.orm.Session,
        alert_data: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
    ):
        notifications_states = self._prepare_notifications_states(alert_data)

        services.api.utils.singletons.db.get_db().store_alert_activation(
            session, alert_data, event_data, notifications_states
        )

    def list_alerts_activations(
        self,
        session: sqlalchemy.orm.Session,
        project: str = "",
    ) -> list[mlrun.common.schemas.AlertActivation]:
        # add filters later
        project = project or mlrun.mlconf.default_project
        return services.api.utils.singletons.db.get_db().list_alerts_activations(
            session, project
        )

    @staticmethod
    def _prepare_notifications_states(
        alert: mlrun.common.schemas.AlertConfig,
    ) -> list[mlrun.common.schemas.NotificationState]:
        # process the notifications associated with the provided alert and construct a list of NotificationState objects
        # each NotificationState represents a unique type of notification (e.g., "slack", "git") and its status,
        # the status will be set to "error" if at least one notification of that type has failed.
        # otherwise, the status will be an empty string if all notifications of that type have succeeded.
        notification_states = {}

        for alert_notification in alert.notifications:
            notification_kind = alert_notification.notification.kind
            notification_status = alert_notification.notification.status

            if notification_kind not in notification_states:
                notification_states[notification_kind] = (
                    "error"
                    if notification_status
                    == mlrun.common.schemas.NotificationStatus.ERROR
                    else ""
                )
            else:
                if notification_status == mlrun.common.schemas.NotificationStatus.ERROR:
                    notification_states[notification_kind] = "error"

        notification_states = [
            mlrun.common.schemas.NotificationState(kind=kind, status=status)
            for kind, status in notification_states.items()
        ]

        return notification_states
