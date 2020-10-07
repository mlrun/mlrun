import logging

import pytest
from sqlalchemy.orm import sessionmaker

from mlrun.api.db.sqldb.models import Schedule

log = logging.getLogger(__name__)


@pytest.fixture
def alembic_config():
    return {
        "script_location": "alembic",
        "before_revision_data": {
            # schedule id migration revision
            "cf21882f938e": [
                {
                    "__tablename__": "schedules_v2",
                    "project": "test-project",
                    "name": "test-schedule1",
                },
                {
                    "__tablename__": "schedules_v2",
                    "project": "test-project",
                    "name": "test-schedule2",
                },
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
    alembic_runner.migrate_up_to("head")

    # schedule id migration revision
    revision_data = alembic_config["before_revision_data"]["cf21882f938e"]

    for index, instance in enumerate(
        alembic_session.query(Schedule).order_by(Schedule.id)
    ):
        assert instance.id == index + 1
        assert instance.name == revision_data[index]["name"]
        assert instance.project == revision_data[index]["project"]
