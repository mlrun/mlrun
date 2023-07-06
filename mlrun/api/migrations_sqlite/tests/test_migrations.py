# Copyright 2018 Iguazio
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
import logging

import pytest
import pytest_alembic.plugin.fixtures
from sqlalchemy.orm import sessionmaker

from mlrun.api.db.sqldb.models import Schedule
from mlrun.config import config

log = logging.getLogger(__name__)


class Constants:
    schedule_table = "schedules_v2"

    schedule_id_revision = "cf21882f938e"
    schedule_id_project = "schedule-id-project"

    last_run_uri_revision = "1c954f8cb32d"
    last_run_uri_project = "last-run-uri-project"

    schedule_concurrency_limit_revision = "e1dd5983c06b"
    schedule_concurrency_limit_project = "schedule-concurrency-limit-project"


alembic_config = {
    "before_revision_data": {
        Constants.schedule_id_revision: [
            {
                "__tablename__": Constants.schedule_table,
                "project": Constants.schedule_id_project,
                "name": name,
            }
            for name in ["test-schedule1", "test-schedule2"]
        ],
        Constants.last_run_uri_revision: [
            {
                "__tablename__": Constants.schedule_table,
                "project": Constants.last_run_uri_project,
                "name": name,
            }
            for name in ["test-schedule3", "test-schedule4"]
        ],
        Constants.schedule_concurrency_limit_revision: [
            {
                "__tablename__": Constants.schedule_table,
                "project": Constants.schedule_concurrency_limit_project,
                "name": name,
            }
            for name in ["test-schedule5", "test-schedule6"]
        ],
    },
}


# alembic modifies the original config for some reason, so in order to
# access it during the tests we need to supply alembic with a copy.
alembic_runner = pytest_alembic.plugin.fixtures.create_alembic_fixture(
    raw_config=copy.deepcopy(alembic_config)
)


@pytest.fixture
def alembic_session(alembic_engine):
    Session = sessionmaker()
    Session.configure(bind=alembic_engine)
    session = Session()
    return session


@pytest.mark.alembic
def test_schedule_id_column(alembic_runner, alembic_session):
    alembic_runner.migrate_up_to(Constants.schedule_id_revision)

    revision_data = alembic_config["before_revision_data"][
        Constants.schedule_id_revision
    ]

    for index, instance in enumerate(
        alembic_session.query(Schedule.id, Schedule.name, Schedule.project)
        .filter_by(project=Constants.schedule_id_project)
        .order_by(Schedule.id)
    ):
        assert instance.id == index + 1
        assert instance.name == revision_data[index]["name"]
        assert instance.project == revision_data[index]["project"]


@pytest.mark.alembic
def test_schedule_last_run_uri_column(alembic_runner, alembic_session):
    alembic_runner.migrate_up_to(Constants.last_run_uri_revision)

    revision_data = alembic_config["before_revision_data"][
        Constants.last_run_uri_revision
    ]

    for index, instance in enumerate(
        alembic_session.query(Schedule.name, Schedule.project, Schedule.last_run_uri)
        .filter_by(project=Constants.last_run_uri_project)
        .order_by(Schedule.id)
    ):
        assert instance.name == revision_data[index]["name"]
        assert instance.project == revision_data[index]["project"]
        assert instance.last_run_uri is None


@pytest.mark.alembic
def test_schedule_concurrency_limit_column(alembic_runner, alembic_session):
    alembic_runner.migrate_up_to(Constants.schedule_concurrency_limit_revision)

    revision_data = alembic_config["before_revision_data"][
        Constants.schedule_concurrency_limit_revision
    ]

    for index, instance in enumerate(
        alembic_session.query(
            Schedule.name, Schedule.project, Schedule.concurrency_limit
        )
        .filter_by(project=Constants.schedule_concurrency_limit_project)
        .order_by(Schedule.id)
    ):
        assert instance.name == revision_data[index]["name"]
        assert instance.project == revision_data[index]["project"]
        assert (
            instance.concurrency_limit
            == config.httpdb.scheduling.default_concurrency_limit
        )
