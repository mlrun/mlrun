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
    ) -> int:
        return framework.utils.singletons.db.get_db().store_alert_activation(
            session, alert_data, event_data
        )

    def update_alert_activation(
        self,
        session,
        activation_id: int,
        activation_time: datetime.datetime,
        number_of_events: Optional[int] = None,
        notifications_states: Optional[
            list[mlrun.common.schemas.NotificationState]
        ] = None,
        update_reset_time: bool = False,
    ):
        framework.utils.singletons.db.get_db().update_alert_activation(
            session=session,
            activation_id=activation_id,
            activation_time=activation_time,
            number_of_events=number_of_events,
            notifications_states=notifications_states,
            update_reset_time=update_reset_time,
        )

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
        offset: Optional[int] = None,
        limit: Optional[int] = None,
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
            offset=offset,
            limit=limit,
        )

    def get_alert_activation(
        self,
        session: sqlalchemy.orm.Session,
        activation_id: int,
    ) -> mlrun.common.schemas.AlertActivation:
        return framework.utils.singletons.db.get_db().get_alert_activation(
            session=session,
            activation_id=activation_id,
        )
