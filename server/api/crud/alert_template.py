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

import mlrun.utils.singleton
import server.api.api.utils
import server.api.utils.helpers
import server.api.utils.singletons.db


class AlertTemplates(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def store_alert_template(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        alert_data: mlrun.common.schemas.AlertTemplate,
    ):
        alert_template = server.api.utils.singletons.db.get_db().get_alert_template(
            session, name
        )

        self._validate_alert_template(alert_data, name)

        if alert_template is not None:
            alert_data.id = alert_template.id

        return server.api.utils.singletons.db.get_db().store_alert_template(
            session, alert_data
        )

    def list_alert_templates(
        self,
        session: sqlalchemy.orm.Session,
    ) -> list[mlrun.common.schemas.AlertTemplate]:
        return server.api.utils.singletons.db.get_db().list_alert_templates(session)

    def get_alert_template(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
    ) -> mlrun.common.schemas.AlertTemplate:
        alert_template = server.api.utils.singletons.db.get_db().get_alert_template(
            session, name
        )
        if alert_template is None:
            raise mlrun.errors.MLRunNotFoundError(f"Alert template {name} not found")

        return alert_template

    def delete_alert_template(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
    ):
        template = server.api.utils.singletons.db.get_db().get_alert_template(
            session, name
        )

        if template is None:
            return

        if template.system_generated:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Cannot delete the Alert template {name}: it is a system template"
            )

        server.api.utils.singletons.db.get_db().delete_alert_template(session, name)

    @staticmethod
    def _validate_alert_template(alert_template, name):
        if (
            alert_template.criteria is not None
            and alert_template.criteria.period is not None
            and server.api.utils.helpers.string_to_timedelta(
                alert_template.criteria.period, raise_on_error=False
            )
            is None
        ):
            raise mlrun.errors.MLRunBadRequestError(
                f"Invalid period ({alert_template.criteria.period}) specified for alert {name}"
            )
