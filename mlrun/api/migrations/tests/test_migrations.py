import logging

import pytest
from sqlalchemy.orm import sessionmaker

from mlrun.api.db.sqldb.models import Schedule

log = logging.getLogger(__name__)


class Constants:
    schedule_table = "schedules_v2"

    schedule_id_revision = "cf21882f938e"
    schedule_id_project = "schedule-id-project"

    last_run_uri_revision = "1c954f8cb32d"
    last_run_uri_project = "last-run-uri-project"


@pytest.fixture
def alembic_config():
    return {
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
        },
    }


@pytest.fixture
def alembic_session(alembic_engine):
    Session = sessionmaker()
    Session.configure(bind=alembic_engine)
    session = Session()
    return session


@pytest.mark.alembic
def test_schedule_id_column(alembic_runner, alembic_session, alembic_config):
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
def test_schedule_last_run_uri_column(alembic_runner, alembic_session, alembic_config):
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
