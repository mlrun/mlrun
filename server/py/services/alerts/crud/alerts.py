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
import typing

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
from mlrun.config import config as mlconfig
from mlrun.utils import logger

import framework.utils.helpers
import framework.utils.lru_cache
import framework.utils.singletons.db
import services.alerts.crud
from framework.utils.notifications.notification_pusher import AlertNotificationPusher


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
        force_reset: bool = False,
    ):
        project = project or mlrun.mlconf.default_project

        existing_alert = framework.utils.singletons.db.get_db().get_alert(
            session, project, name
        )

        self._validate_alert(alert_data, name, project)

        if alert_data.criteria is None:
            alert_data.criteria = mlrun.common.schemas.alert.AlertCriteria()

        if existing_alert is not None:
            self._delete_notifications(existing_alert)
            self._get_alert_by_id_cached().cache_remove(session, existing_alert.id)

            for kind in existing_alert.trigger.events:
                services.alerts.crud.Events().remove_event_configuration(
                    project, kind, existing_alert.id
                )

            # preserve the original creation time and id of the alert so that modifying the alert does not change them
            alert_data.created = existing_alert.created
            alert_data.id = existing_alert.id

            # set the updated field to reflect the latest modification time of the alert
            alert_data.updated = mlrun.utils.now_date()

            old_state = framework.utils.singletons.db.get_db().get_alert_state_dict(
                session, existing_alert.id
            )
            existing_alert.state = (
                mlrun.common.schemas.AlertActiveState.ACTIVE
                if old_state["active"]
                else mlrun.common.schemas.AlertActiveState.INACTIVE
            )
        else:
            num_alerts = (
                framework.utils.singletons.db.get_db().get_num_configured_alerts(
                    session
                )
            )
            if num_alerts >= mlconfig.alerts.max_allowed:
                raise mlrun.errors.MLRunPreconditionFailedError(
                    f"Allowed number of alerts exceeded: {num_alerts}"
                )

        self._validate_and_mask_notifications(alert_data)

        new_alert = framework.utils.singletons.db.get_db().store_alert(
            session, alert_data
        )

        for kind in new_alert.trigger.events:
            services.alerts.crud.Events().add_event_configuration(
                project, kind, new_alert.id
            )

        # if the alert already exists we should check if it should be reset or not
        if existing_alert is not None:
            should_reset, reset_reason = self._should_reset_alert(
                existing_alert, alert_data, force_reset
            )
            if should_reset:
                logger.debug(
                    "Resetting alert before storing",
                    project=project,
                    alert_name=name,
                    reason=reset_reason,
                )
                self.reset_alert(session, project, new_alert.name)

        framework.utils.singletons.db.get_db().enrich_alert(session, new_alert)

        logger.debug("Stored alert", alert=new_alert)

        return new_alert

    def list_alerts(
        self,
        session: sqlalchemy.orm.Session,
        project: typing.Optional[typing.Union[str, list[str]]] = None,
    ) -> list[mlrun.common.schemas.AlertConfig]:
        project = project or mlrun.mlconf.default_project
        return framework.utils.singletons.db.get_db().list_alerts(session, project)

    def get_enriched_alert(
        self, session: sqlalchemy.orm.Session, project: str, name: str
    ):
        alert = framework.utils.singletons.db.get_db().get_alert(session, project, name)
        if alert is None:
            raise mlrun.errors.MLRunNotFoundError(
                f"Alert {name} for project {project} not found"
            )

        framework.utils.singletons.db.get_db().enrich_alert(session, alert)
        return alert

    def get_alert(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        name: str,
    ) -> mlrun.common.schemas.AlertConfig:
        project = project or mlrun.mlconf.default_project
        return framework.utils.singletons.db.get_db().get_alert(session, project, name)

    def delete_alert(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        name: str,
    ):
        project = project or mlrun.mlconf.default_project

        alert = framework.utils.singletons.db.get_db().get_alert(session, project, name)

        if alert is None:
            return

        for kind in alert.trigger.events:
            services.alerts.crud.Events().remove_event_configuration(
                project, kind, alert.id
            )

        framework.utils.singletons.db.get_db().delete_alert(session, project, name)
        self._clear_alert_states(alert)

    def process_event(
        self,
        session: sqlalchemy.orm.Session,
        alert_id: int,
        event_data: mlrun.common.schemas.Event,
    ):
        state = self._get_alert_state_cached()(session, alert_id)
        alert = self._get_alert_by_id_cached()(session, alert_id)

        # check if the entity of the alert matches the one in event
        if not self._event_entity_matches(alert.entities, event_data.entity):
            return

        state_obj = self._states.get(alert.id, {"event_timestamps": []})
        state_obj["event_timestamps"].append(event_data.timestamp)

        # Exit early if state is active (no further processing needed)
        if state["active"]:
            self._states[alert.id] = state_obj
            return

        send_notification = self._should_send_notification(alert, state_obj)
        update_state_cache = True
        if send_notification:
            update_state_cache = self._handle_notification(
                session, alert, state, state_obj, event_data
            )

        if update_state_cache:
            # we don't want to update the state if reset_alert() was called, as we will override the reset
            self._states[alert.id] = state_obj

    def populate_event_cache(self, session: sqlalchemy.orm.Session):
        try:
            self._try_populate_event_cache(session)
        except Exception as exc:
            logger.error(
                "Error populating event cache. Transitioning state to offline!",
                exc=mlrun.errors.err_to_str(exc),
            )
            mlconfig.httpdb.state = mlrun.common.schemas.APIStates.offline
            return

        services.alerts.crud.Events().cache_initialized = True
        logger.debug("Finished populating event cache")

    def _should_send_notification(
        self, alert: mlrun.common.schemas.AlertConfig, state_obj: dict
    ) -> bool:
        if alert.criteria.period:
            offset = self._get_event_offset(alert)
            self._normalize_events(
                state_obj,
                framework.utils.helpers.string_to_timedelta(
                    alert.criteria.period, offset, raise_on_error=False
                ),
            )
        return len(state_obj["event_timestamps"]) >= alert.criteria.count

    def _get_number_of_events(self, alert_id: int) -> int:
        state_obj = self._states.get(alert_id, {"event_timestamps": []})
        return len(state_obj["event_timestamps"])

    @staticmethod
    def _get_event_offset(alert: mlrun.common.schemas.AlertConfig) -> int:
        if alert.entities.kind == mlrun.common.schemas.alert.EventEntityKind.JOB:
            return int(mlconfig.monitoring.runs.interval)
        return 0

    def _handle_notification(
        self,
        session: sqlalchemy.orm.Session,
        alert: mlrun.common.schemas.AlertConfig,
        state: dict,
        state_obj: dict,
        event_data: mlrun.common.schemas.Event,
    ) -> bool:
        keep_cache = True
        active = False
        state["count"] += 1

        if alert.reset_policy == "auto":
            self.reset_alert(session, alert.project, alert.name)
            keep_cache = False
        activation_id = services.alerts.crud.AlertActivation().store_alert_activation(
            session, alert, event_data
        )

        if alert.reset_policy == "manual":
            active = True
            state["active"] = True
            state_obj["last_activation_id"] = activation_id

        logger.debug("Sending notifications for alert", name=alert.name)
        AlertNotificationPusher().push(
            alert,
            event_data,
            activation_id=activation_id,
            activation_time=event_data.timestamp,
        )

        framework.utils.singletons.db.get_db().store_alert_state(
            session,
            alert.project,
            alert.name,
            count=state["count"],
            last_updated=event_data.timestamp,
            obj=state_obj,
            active=active,
        )
        return keep_cache

    @classmethod
    def _get_alert_by_id_cached(cls):
        if not cls._alert_cache:
            cls._alert_cache = framework.utils.lru_cache.LRUCache(
                framework.utils.singletons.db.get_db().get_alert_by_id,
                maxsize=1000,
                ignore_args_for_hash=[0],
            )

        return cls._alert_cache

    @classmethod
    def _get_alert_state_cached(cls):
        if not cls._alert_state_cache:
            cls._alert_state_cache = framework.utils.lru_cache.LRUCache(
                framework.utils.singletons.db.get_db().get_alert_state_dict,
                maxsize=1000,
                ignore_args_for_hash=[0],
            )
        return cls._alert_state_cache

    @staticmethod
    def _try_populate_event_cache(session: sqlalchemy.orm.Session):
        for alert in framework.utils.singletons.db.get_db().get_all_alerts(session):
            for event_name in alert.trigger.events:
                services.alerts.crud.Events().add_event_configuration(
                    alert.project, event_name, alert.id
                )

    def process_event_no_cache(
        self,
        session: sqlalchemy.orm.Session,
        event_name: str,
        event_data: mlrun.common.schemas.Event,
    ):
        for alert in framework.utils.singletons.db.get_db().get_all_alerts(session):
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
                and framework.utils.helpers.string_to_timedelta(
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
                    f"for alert {name} for project {project}"
                )
            notification_object = mlrun.model.Notification.from_dict(
                alert_notification.notification.dict()
            )
            notification_object.validate_notification()
            if (
                alert_notification.cooldown_period is not None
                and framework.utils.helpers.string_to_timedelta(
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
        events = obj["event_timestamps"]
        for event in events:
            if isinstance(event, str):
                event_time = datetime.datetime.fromisoformat(event)
            else:
                event_time = event
            if now > event_time + period:
                events.remove(event)

    def reset_alert(self, session: sqlalchemy.orm.Session, project: str, name: str):
        alert = framework.utils.singletons.db.get_db().get_alert(session, project, name)
        if alert is None:
            raise mlrun.errors.MLRunNotFoundError(
                f"Alert {name} for project {project} does not exist"
            )

        if alert.reset_policy == mlrun.common.schemas.alert.ResetPolicy.MANUAL:
            self._update_alert_activation_on_reset(
                session=session,
                project=project,
                alert=alert,
            )
        framework.utils.singletons.db.get_db().store_alert_state(
            session, project, name, last_updated=None
        )
        self._get_alert_state_cached().cache_remove(session, alert.id)
        self._clear_alert_states(alert)

    def _update_alert_activation_on_reset(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        alert: mlrun.common.schemas.AlertConfig,
    ) -> None:
        alert_state = framework.utils.singletons.db.get_db().get_alert_state_dict(
            session, alert.id
        )
        if not alert_state:
            logger.warning(
                "No alert state found for alert, skipping activation update on reset",
                project=project,
                alert_name=alert.name,
            )
            return

        # update number_of_events if required
        number_of_events = self._get_number_of_events(alert.id)
        activation_time = alert_state.get("last_updated")

        # or {} is needed of the case if full_object is None
        activation_id = (alert_state.get("full_object") or {}).get("last_activation_id")
        if activation_time and activation_id:
            framework.utils.singletons.db.get_db().update_alert_activation(
                session,
                activation_id=activation_id,
                activation_time=activation_time,
                # If they are equal, the number_of_events is already set to the correct value.
                # Additionally, this ensures safety by avoiding potential cache issues.
                # For example, if the service restarts, we might lose all information about the number of events.
                number_of_events=number_of_events
                if number_of_events > alert.criteria.count
                else None,
                update_reset_time=True,
            )
        else:
            logger.warning(
                "No activation id or last activation time found for alert, skipping activation update on reset",
                project=project,
                alert_name=alert.name,
                activation_id=activation_id,
                activation_time=activation_time,
            )

    @staticmethod
    def _should_reset_alert(old_alert_data, alert_data, force_reset):
        if force_reset:
            return True, "force_reset being True"

        # reset the alert if the policy was modified from manual to auto while the state is active
        old_reset_policy = getattr(old_alert_data, "reset_policy")
        new_reset_policy = getattr(alert_data, "reset_policy")
        if (
            old_alert_data.state == mlrun.common.schemas.AlertActiveState.ACTIVE
            and old_reset_policy == mlrun.common.schemas.alert.ResetPolicy.MANUAL
            and new_reset_policy == mlrun.common.schemas.alert.ResetPolicy.AUTO
        ):
            return True, "reset-policy changed from manual to auto"

        # reset the alert if a functional parameter (entities, trigger, or criteria) has changed, as these affect the
        # conditions for alert activation.
        functional_parameters = ["entities", "trigger", "criteria"]
        for attr in functional_parameters:
            if getattr(old_alert_data, attr) != getattr(alert_data, attr):
                return True, f"changes in {attr}"

        return False, None

    @staticmethod
    def _delete_notifications(alert: mlrun.common.schemas.AlertConfig):
        for notification in alert.notifications:
            framework.utils.notifications.delete_notification_params_secret(
                alert.project, notification.notification
            )

    @staticmethod
    def _validate_and_mask_notifications(alert_data):
        notifications = [
            mlrun.common.schemas.notification.Notification(**notification.to_dict())
            for notification in framework.utils.notifications.validate_and_mask_notification_list(
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
