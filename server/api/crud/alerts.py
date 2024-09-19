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
import re

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.api.utils
import server.api.utils.helpers
import server.api.utils.lru_cache
import server.api.utils.singletons.db
from mlrun.config import config as mlconfig
from mlrun.utils import logger
from server.api.utils.notification_pusher import AlertNotificationPusher


class Alerts(
    metaclass=mlrun.utils.singleton.Singleton,
):
    _states = dict()
    _alert_cache = None
    _alert_state_cache = None

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

        if alert_data.criteria is None:
            alert_data.criteria = mlrun.common.schemas.alert.AlertCriteria()

        if alert is not None:
            self._delete_notifications(alert)
            self._get_alert_by_id_cached().cache_remove(session, alert.id)
        else:
            num_alerts = (
                server.api.utils.singletons.db.get_db().get_num_configured_alerts(
                    session
                )
            )
            if num_alerts >= mlconfig.alerts.max_allowed:
                raise mlrun.errors.MLRunPreconditionFailedError(
                    f"Allowed number of alerts exceeded: {num_alerts}"
                )

        self._validate_and_mask_notifications(alert_data)

        if alert is not None:
            for kind in alert.trigger.events:
                server.api.crud.Events().remove_event_configuration(
                    project, kind, alert.id
                )
            alert_data.created = alert.created
            alert_data.id = alert.id

        new_alert = server.api.utils.singletons.db.get_db().store_alert(
            session, alert_data
        )

        for kind in new_alert.trigger.events:
            server.api.crud.Events().add_event_configuration(
                project, kind, new_alert.id
            )

        self.reset_alert(session, project, new_alert.name)

        server.api.utils.singletons.db.get_db().enrich_alert(session, new_alert)

        logger.debug("Stored alert", alert=new_alert)

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
            server.api.crud.Events().remove_event_configuration(project, kind, alert.id)

        server.api.utils.singletons.db.get_db().delete_alert(session, project, name)
        self._clear_alert_states(alert)

    def process_event(
        self,
        session: sqlalchemy.orm.Session,
        alert_id: int,
        event_data: mlrun.common.schemas.Event,
    ):
        state = self._get_alert_state_cached()(session, alert_id)
        if state["active"]:
            return

        alert = self._get_alert_by_id_cached()(session, alert_id)

        state_obj = None
        # check if the entity of the alert matches the one in event
        if self._event_entity_matches(alert.entities, event_data.entity):
            send_notification = False

            if alert.criteria is not None:
                state_obj = self._states.get(alert.id, {"events": []})
                state_obj["events"].append(event_data.timestamp)

                if alert.criteria.period is not None:
                    # adjust the sliding window of events
                    # in case the EventEntityKind is JOB then we should consider the runs monitoring interval here
                    # because the monitoring runs might miss events occurring just before the interval.
                    offset = 0
                    if (
                        alert.entities.kind
                        == mlrun.common.schemas.alert.EventEntityKind.JOB
                    ):
                        offset = int(mlconfig.monitoring.runs.interval)
                    self._normalize_events(
                        state_obj,
                        server.api.utils.helpers.string_to_timedelta(
                            alert.criteria.period, offset, raise_on_error=False
                        ),
                    )

                if len(state_obj["events"]) >= alert.criteria.count:
                    send_notification = True
            else:
                send_notification = True

            active = False
            update_state = True
            if send_notification:
                state["count"] += 1
                logger.debug("Sending notifications for alert", name=alert.name)
                AlertNotificationPusher().push(alert, event_data)

                if alert.reset_policy == "auto":
                    self.reset_alert(session, alert.project, alert.name)
                    update_state = False
                else:
                    active = True
                    state["active"] = True
                    self._get_alert_state_cached().cache_replace(
                        state, session, alert.id
                    )

                # we store the state along with the events that triggered the alert
                server.api.utils.singletons.db.get_db().store_alert_state(
                    session,
                    alert.project,
                    alert.name,
                    count=state["count"],
                    last_updated=event_data.timestamp,
                    obj=state_obj,
                    active=active,
                )

            if update_state:
                # we don't want to update the state if reset_alert() was called, as we will override the reset
                self._states[alert.id] = state_obj

    def populate_event_cache(self, session: sqlalchemy.orm.Session):
        try:
            self._try_populate_event_cache(session)
        except Exception as exc:
            logger.error(
                "Error populating event cache for alerts. Transitioning state to offline!",
                exc=mlrun.errors.err_to_str(exc),
            )
            mlconfig.httpdb.state = mlrun.common.schemas.APIStates.offline
            return

        server.api.crud.Events().cache_initialized = True
        logger.debug("Finished populating event cache for alerts")

    @classmethod
    def _get_alert_by_id_cached(cls):
        if not cls._alert_cache:
            cls._alert_cache = server.api.utils.lru_cache.LRUCache(
                server.api.utils.singletons.db.get_db().get_alert_by_id,
                maxsize=1000,
                ignore_args_for_hash=[0],
            )

        return cls._alert_cache

    @classmethod
    def _get_alert_state_cached(cls):
        if not cls._alert_state_cache:
            cls._alert_state_cache = server.api.utils.lru_cache.LRUCache(
                server.api.utils.singletons.db.get_db().get_alert_state_dict,
                maxsize=1000,
                ignore_args_for_hash=[0],
            )
        return cls._alert_state_cache

    @staticmethod
    def _try_populate_event_cache(session: sqlalchemy.orm.Session):
        for alert in server.api.utils.singletons.db.get_db().get_all_alerts(session):
            for event_name in alert.trigger.events:
                server.api.crud.Events().add_event_configuration(
                    alert.project, event_name, alert.id
                )

    def process_event_no_cache(
        self,
        session: sqlalchemy.orm.Session,
        event_name: str,
        event_data: mlrun.common.schemas.Event,
    ):
        for alert in server.api.utils.singletons.db.get_db().get_all_alerts(session):
            for config_event_name in alert.trigger.events:
                if config_event_name == event_name:
                    self.process_event(session, alert.id, event_data)

    @staticmethod
    def _event_entity_matches(alert_entity, event_entity):
        if "*" in alert_entity.ids:
            return True

        if event_entity.ids[0] in alert_entity.ids:
            return True

        return False

    def _validate_alert(self, alert, name, project):
        self.validate_alert_name(alert.name)
        if name != alert.name:
            raise mlrun.errors.MLRunBadRequestError(
                f"Alert name mismatch for alert {name} for project {project}. Provided {alert.name}"
            )

        if alert.criteria is not None:
            if alert.criteria.count >= mlconfig.alerts.max_criteria_count:
                raise mlrun.errors.MLRunPreconditionFailedError(
                    f"Maximum criteria count exceeded: {alert.criteria.count}"
                )

            if (
                alert.criteria.period is not None
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
    def validate_alert_name(name: str) -> None:
        if not re.fullmatch(r"^[a-zA-Z0-9-]+$", name):
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid alert name '{name}'. Alert names can only contain alphanumeric characters and hyphens."
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
        self._get_alert_state_cached().cache_remove(session, alert.id)
        self._clear_alert_states(alert)

    @staticmethod
    def _delete_notifications(alert: mlrun.common.schemas.AlertConfig):
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

    def _clear_alert_states(self, alert):
        if alert.id in self._states:
            self._states.pop(alert.id)
