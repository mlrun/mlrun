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

import framework.utils.singletons.db


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

        framework.utils.singletons.db.get_db().store_alert_activation(
            session, alert_data, event_data, notifications_states
        )

    @staticmethod
    def _prepare_notifications_states(
        alert: mlrun.common.schemas.AlertConfig,
    ) -> list[mlrun.common.schemas.NotificationState]:
        # process the notifications associated with the provided alert and construct a list of NotificationState objects
        # each NotificationState represents a unique type of notification (e.g., "slack", "git") and its status.
        # if at least one notification of that type failed, the error message from the most recent failure will
        # be stored. if no failure occurred, the status will be an empty string indicating success.
        notification_states = [
            mlrun.common.schemas.NotificationState(
                kind=alert_notification.notification.kind,
                err=alert_notification.notification.reason,
            )
            for alert_notification in alert.notifications
        ]
        return notification_states
