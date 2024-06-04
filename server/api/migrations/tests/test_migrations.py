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
import copy
import json
import logging
import pathlib

import pytest
import pytest_alembic.plugin.fixtures
import sqlalchemy
from pytest_alembic.tests import (  # noqa
    test_model_definitions_match_ddl,
    test_single_head_revision,
    test_up_down_consistency,
    test_upgrade,
)
from sqlalchemy.orm import sessionmaker

import mlrun
from server.api.db.sqldb.models import Run

log = logging.getLogger(__name__)


class Constants:
    ini_file_path = str(
        pathlib.Path(__file__).absolute().parent.parent.parent / "alembic.ini"
    )
    notifications_table = "runs_notifications"

    notifications_params_to_secret_params_revision = "eefc169f7633"
    notifications_params_to_secret_params_project = (
        "notifications_params_to_secret_params_project"
    )


alembic_config = {
    "file": Constants.ini_file_path,
    "before_revision_data": {
        Constants.notifications_params_to_secret_params_revision: [
            {
                "__tablename__": Constants.notifications_table,
                "project": Constants.notifications_params_to_secret_params_project,
                "name": name,
                "kind": "console",
                "message": "test",
                "severity": "info",
                "when": "completed",
                "params": json.dumps({"obj": {"x": 99}}),
                "condition": "",
                "status": "",
            }
            for name in ["notifications1"]
        ],
    },
}


# alembic modifies the original config for some reason, so in order to
# access it during the tests we need to supply alembic with a copy.
alembic_runner = pytest_alembic.plugin.fixtures.create_alembic_fixture(
    raw_config=copy.deepcopy(alembic_config)
)


@pytest.fixture
def alembic_engine():
    return sqlalchemy.create_engine(mlrun.mlconf.httpdb.dsn)


@pytest.fixture
def alembic_session(alembic_engine):
    session_maker = sessionmaker()
    session_maker.configure(bind=alembic_engine)
    session = session_maker()
    return session


@pytest.mark.alembic
def test_notification_params_to_secret_params(alembic_runner, alembic_session):
    alembic_runner.migrate_up_to(
        Constants.notifications_params_to_secret_params_revision
    )

    revision_data = alembic_config["before_revision_data"][
        Constants.notifications_params_to_secret_params_revision
    ]

    for index, item in enumerate(
        alembic_session.query(Run.Notification.params, Run.Notification.secret_params)
        .filter_by(project=Constants.notifications_params_to_secret_params_project)
        .order_by(Run.Notification.id)
    ):
        assert not item.params
        assert item.secret_params == revision_data[index]["params"]
