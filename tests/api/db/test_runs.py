import pytest
from mlrun.config import config
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.models import Run
from tests.api.db.conftest import dbs


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_runs_name_filter(db: DBInterface, db_session: Session):
    run_name_1 = "run_name_1"
    run_name_2 = "run_name_2"
    run_1 = {"metadata": {"name": run_name_1}, "status": {"bla": "blabla"}}
    run_2 = {"metadata": {"name": run_name_2}, "status": {"bla": "blabla"}}
    run_uid_1 = "run_uid_1"
    run_uid_2 = "run_uid_2"

    db.store_run(
        db_session, run_1, run_uid_1,
    )
    db.store_run(
        db_session, run_2, run_uid_2,
    )
    runs = db.list_runs(db_session)
    assert len(runs) == 2

    runs = db.list_runs(db_session, name=run_name_1)
    assert len(runs) == 1
    assert runs[0]["metadata"]["name"] == run_name_1

    runs = db.list_runs(db_session, name=run_name_2)
    assert len(runs) == 1
    assert runs[0]["metadata"]["name"] == run_name_2

    runs = db.list_runs(db_session, name="run_name")
    assert len(runs) == 2


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_runs_state_filter(db: DBInterface, db_session: Session):
    run_without_state_uid = "run_without_state_uid"
    run_without_state = {"metadata": {"uid": run_without_state_uid}, "bla": "blabla"}
    db.store_run(
        db_session, run_without_state, run_without_state_uid,
    )

    run_with_json_state_state = "some_json_state"
    run_with_json_state_uid = "run_with_json_state_uid"
    run_with_json_state = {
        "metadata": {"uid": run_with_json_state_uid},
        "status": {"state": run_with_json_state_state},
    }
    run = Run(
        uid=run_with_json_state_uid,
        project=config.default_project,
        iteration=0,
        start_time=datetime.now(timezone.utc),
    )
    run.struct = run_with_json_state
    db._upsert(db_session, run, ignore=True)

    run_with_record_state_state = "some_record_state"
    run_with_record_state_uid = "run_with_record_state_uid"
    run_with_record_state = {
        "metadata": {"uid": run_with_record_state_uid},
        "bla": "blabla",
    }
    run = Run(
        uid=run_with_record_state_uid,
        project=config.default_project,
        iteration=0,
        state=run_with_record_state_state,
        start_time=datetime.now(timezone.utc),
    )
    run.struct = run_with_record_state
    db._upsert(db_session, run, ignore=True)

    run_with_equal_json_and_record_state_state = "some_equal_json_and_record_state"
    run_with_equal_json_and_record_state_uid = (
        "run_with_equal_json_and_record_state_uid"
    )
    run_with_equal_json_and_record_state = {
        "metadata": {"uid": run_with_equal_json_and_record_state_uid},
        "status": {"state": run_with_equal_json_and_record_state_state},
    }
    db.store_run(
        db_session,
        run_with_equal_json_and_record_state,
        run_with_equal_json_and_record_state_uid,
    )

    run_with_unequal_json_and_record_state_json_state = "some_unequal_json_state"
    run_with_unequal_json_and_record_state_record_state = "some_unequal_record_state"
    run_with_unequal_json_and_record_state_uid = (
        "run_with_unequal_json_and_record_state_uid"
    )
    run_with_unequal_json_and_record_state = {
        "metadata": {"uid": run_with_unequal_json_and_record_state_uid},
        "status": {"state": run_with_unequal_json_and_record_state_json_state},
    }
    run = Run(
        uid=run_with_unequal_json_and_record_state_uid,
        project=config.default_project,
        iteration=0,
        state=run_with_unequal_json_and_record_state_record_state,
        start_time=datetime.now(timezone.utc),
    )
    run.struct = run_with_unequal_json_and_record_state
    db._upsert(db_session, run, ignore=True)

    runs = db.list_runs(db_session)
    assert len(runs) == 5

    runs = db.list_runs(db_session, state="some_")
    assert len(runs) == 4
    assert run_without_state_uid not in [run["metadata"]["uid"] for run in runs]

    runs = db.list_runs(db_session, state=run_with_json_state_state)
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_json_state_uid

    runs = db.list_runs(db_session, state=run_with_record_state_state)
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_record_state_uid

    runs = db.list_runs(db_session, state=run_with_equal_json_and_record_state_state)
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_equal_json_and_record_state_uid

    runs = db.list_runs(
        db_session, state=run_with_unequal_json_and_record_state_json_state
    )
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_unequal_json_and_record_state_uid

    runs = db.list_runs(
        db_session, state=run_with_unequal_json_and_record_state_record_state
    )
    assert len(runs) == 0
