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

import datetime

import sqlalchemy.orm

import mlrun.utils.singleton
import server.api.api.utils
import server.api.utils.helpers
import server.api.utils.singletons.db
from mlrun.utils import logger
from server.api.utils.notification_pusher import AlertNotificationPusher


class Alerts(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_alert(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        alert_data: mlrun.common.schemas.AlertConfig,
    ):
        project = project or mlrun.mlconf.default_project

        alert = server.api.utils.singletons.db.get_db().get_alert(
            session, project, name
        )

        self._validate_alert(alert_data, name, project)

        if alert is not None:
            self._delete_notifications(alert)

        self._validate_and_mask_notifications(alert_data)

        if alert is not None:
            for kind in alert.trigger.events:
                server.api.crud.Events().remove_event_configuration(project, kind)
            alert_data.created = alert.created
            alert_data.id = alert.id

        new_alert = server.api.utils.singletons.db.get_db().store_alert(
            session, alert_data
        )

        for kind in new_alert.trigger.events:
            server.api.crud.Events().add_event_configuration(
                project, kind, new_alert.name
            )

        self.reset_alert(session, project, new_alert.name)

        server.api.utils.singletons.db.get_db().enrich_alert(session, new_alert)

        return new_alert

    def list_alerts(
        self,
        session: sqlalchemy.orm.Session,
        project: str = "",
    ) -> list[mlrun.common.schemas.AlertConfig]:
        project = project or mlrun.mlconf.default_project
        return server.api.utils.singletons.db.get_db().list_alerts(session, project)

    def get_enriched_alert(
        self, session: sqlalchemy.orm.Session, project: str, name: str
    ):
        alert = server.api.utils.singletons.db.get_db().get_alert(
            session, project, name
        )
        if alert is None:
            raise mlrun.errors.MLRunNotFoundError(
                f"Alert {name} for project {project} not found"
            )

        server.api.utils.singletons.db.get_db().enrich_alert(session, alert)
        return alert

    def get_alert(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        name: str,
    ) -> mlrun.common.schemas.AlertConfig:
        project = project or mlrun.mlconf.default_project
        return server.api.utils.singletons.db.get_db().get_alert(session, project, name)

    def delete_alert(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        name: str,
    ):
        project = project or mlrun.mlconf.default_project

        alert = server.api.utils.singletons.db.get_db().get_alert(
            session, project, name
        )

        if alert is None:
            return

        for kind in alert.trigger.events:
            server.api.crud.Events().remove_event_configuration(project, kind)

        server.api.utils.singletons.db.get_db().delete_alert(session, project, name)

    def process_event(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        name: str,
        event_data: mlrun.common.schemas.Event,
    ):
        alert = server.api.utils.singletons.db.get_db().get_alert(
            session, project, name
        )

        state = server.api.utils.singletons.db.get_db().get_alert_state(
            session, alert.id
        )
        if state.active:
            logger.debug("Alert already active, so ignoring event", name=alert.name)
            return

        state_obj = None
        # check if the entity of the alert matches the one in event
        if self._event_entity_matches(alert.entities, event_data.entity):
            send_notification = False

            if alert.criteria is not None:
                state_obj = state.full_object

                if state_obj is None:
                    state_obj = {"events": [event_data.timestamp]}

                if alert.criteria.period is not None:
                    state_obj["events"].append(event_data.timestamp)
                    # adjust the sliding window of events
                    self._normalize_events(
                        state_obj,
                        server.api.utils.helpers.string_to_timedelta(
                            alert.criteria.period, raise_on_error=False
                        ),
                    )

                if len(state_obj["events"]) >= alert.criteria.count:
                    send_notification = True
            else:
                send_notification = True

            active = False
            if send_notification:
                state.count += 1
                logger.debug("Sending notifications for alert", name=alert.name)
                AlertNotificationPusher().push(alert, event_data)

                if alert.reset_policy == "auto":
                    self.reset_alert(session, alert.project, alert.name)
                else:
                    active = True

            server.api.utils.singletons.db.get_db().store_alert_state(
                session,
                alert.project,
                alert.name,
                count=state.count,
                last_updated=event_data.timestamp,
                obj=state_obj,
                active=active,
            )

    @staticmethod
    def _event_entity_matches(alert_entity, event_entity):
        if "*" in alert_entity.ids:
            return True

        if event_entity.ids[0] in alert_entity.ids:
            return True

        return False

    @staticmethod
    def _validate_alert(alert, name, project):
        if name != alert.name:
            raise mlrun.errors.MLRunBadRequestError(
                f"Alert name mismatch for alert {name} for project {project}. Provided {alert.name}"
            )

        if (
            alert.criteria is not None
            and alert.criteria.period is not None
            and server.api.utils.helpers.string_to_timedelta(
                alert.criteria.period, raise_on_error=False
            )
            is None
        ):
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid period ({alert.criteria.period}) specified for alert {name} for project {project}"
            )

        for alert_notification in alert.notifications:
            if alert_notification.notification.kind not in [
                mlrun.common.schemas.NotificationKind.git,
                mlrun.common.schemas.NotificationKind.slack,
                mlrun.common.schemas.NotificationKind.webhook,
            ]:
                raise mlrun.errors.MLRunBadRequestError(
                    f"Unsupported notification ({alert_notification.notification.kind}) "
                    "for alert {name} for project {project}"
                )
            notification_object = mlrun.model.Notification.from_dict(
                alert_notification.notification.dict()
            )
            notification_object.validate_notification()
            if (
                alert_notification.cooldown_period is not None
                and server.api.utils.helpers.string_to_timedelta(
                    alert_notification.cooldown_period, raise_on_error=False
                )
                is None
            ):
                raise mlrun.errors.MLRunBadRequestError(
                    f"Invalid cooldown_period ({alert_notification.cooldown_period}) "
                    "specified for alert {name} for project {project}"
                )

        if alert.entities.project != project:
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid alert entity project ({alert.entities.project}) for alert {name} for project {project}"
            )

    @staticmethod
    def _normalize_events(obj, period):
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        events = obj["events"]
        for event in events:
            if isinstance(event, str):
                event_time = datetime.datetime.fromisoformat(event)
            else:
                event_time = event
            if now > event_time + period:
                events.remove(event)

    def reset_alert(self, session: sqlalchemy.orm.Session, project: str, name: str):
        alert = server.api.utils.singletons.db.get_db().get_alert(
            session, project, name
        )
        if alert is None:
            raise mlrun.errors.MLRunNotFoundError(
                f"Alert {name} for project {project} does not exist"
            )

        server.api.utils.singletons.db.get_db().store_alert_state(
            session, project, name, last_updated=None
        )

    def _delete_notifications(self, alert: mlrun.common.schemas.AlertConfig):
        for notification in alert.notifications:
            server.api.api.utils.delete_notification_params_secret(
                alert.project, notification.notification
            )

    @staticmethod
    def _validate_and_mask_notifications(alert_data):
        notifications = [
            mlrun.common.schemas.notification.Notification(**notification.to_dict())
            for notification in server.api.api.utils.validate_and_mask_notification_list(
                alert_data.get_raw_notifications(), None, alert_data.project
            )
        ]
        cooldowns = [
            notification.cooldown_period for notification in alert_data.notifications
        ]

        alert_data.notifications = [
            mlrun.common.schemas.alert.AlertNotification(
                cooldown_period=cooldown, notification=notification
            )
            for cooldown, notification in zip(cooldowns, notifications)
        ]
