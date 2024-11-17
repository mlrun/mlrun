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
import sqlalchemy.orm

import framework.constants
import framework.db.session
import framework.db.sqldb.db
import services.alerts.crud
from framework.db.session import close_session, create_session


def update_default_configuration_data(logger):
    logger.debug("Updating default configuration data")
    framework.db.session.run_function_with_new_db_session(
        services.alerts.crud.Alerts().populate_event_cache
    )
    db_session = create_session()
    try:
        db = framework.db.sqldb.db.SQLDB()
        _add_default_alert_templates(db, db_session)
    finally:
        close_session(db_session)


def _add_default_alert_templates(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    for template in framework.constants.pre_defined_templates:
        record = db.get_alert_template(db_session, template.template_name)
        if record is None or record.templates_differ(template):
            db.store_alert_template(db_session, template)
