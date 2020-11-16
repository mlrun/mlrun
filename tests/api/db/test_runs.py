import time
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_runs_times_filters(db: DBInterface, db_session: Session):

    timestamp1 = datetime.now(timezone.utc)

    time.sleep(1)

    timestamp2 = datetime.now(timezone.utc)

    normal_run_1_uid = "normal_run_1_uid"
    normal_run_1 = {
        "metadata": {"uid": normal_run_1_uid},
        "status": {"last_update": timestamp2.isoformat()},
    }
    run = Run(
        uid=normal_run_1_uid,
        project=config.default_project,
        iteration=0,
        start_time=timestamp1,
    )
    run.struct = normal_run_1
    db._upsert(db_session, run, ignore=True)

    timestamp3 = datetime.now(timezone.utc)

    run_without_last_update_uid = "run_without_last_update_uid"
    run_without_last_update = {
        "metadata": {"uid": run_without_last_update_uid},
        "status": {},
    }
    run = Run(
        uid=run_without_last_update_uid,
        project=config.default_project,
        iteration=0,
        start_time=timestamp3,
    )
    run.struct = run_without_last_update
    db._upsert(db_session, run, ignore=True)

    timestamp4 = datetime.now(timezone.utc)

    time.sleep(1)

    timestamp5 = datetime.now(timezone.utc)

    normal_run_2_uid = "normal_run_2_uid"
    normal_run_2 = {
        "metadata": {"uid": normal_run_2_uid},
        "status": {"last_update": timestamp5.isoformat()},
    }
    run = Run(
        uid=normal_run_2_uid,
        project=config.default_project,
        iteration=0,
        start_time=timestamp4,
    )
    run.struct = normal_run_2
    db._upsert(db_session, run, ignore=True)

    runs = db.list_runs(db_session)
    assert len(runs) == 3

    # all start time range
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp1,
        start_time_to=timestamp5,
    )
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp1,
    )
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid],
        start_time_to=timestamp5,
    )

    # all last update time range (shouldn't contain run_without_last_update)
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid, normal_run_2_uid],
        last_update_time_from=timestamp1,
        last_update_time_to=timestamp5,
    )
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid, normal_run_2_uid],
        last_update_time_from=timestamp1,
    )
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid, normal_run_2_uid],
        last_update_time_to=timestamp5,
    )

    # catch only first
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid],
        start_time_from=timestamp1,
        start_time_to=timestamp2,
    )
    assert_time_range(
        db, db_session, [normal_run_1_uid], start_time_to=timestamp2,
    )
    assert_time_range(
        db,
        db_session,
        [normal_run_1_uid],
        last_update_time_from=timestamp1,
        last_update_time_to=timestamp2,
    )

    # catch run_without_last_update and last
    assert_time_range(
        db,
        db_session,
        [normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp3,
        start_time_to=timestamp5,
    )
    assert_time_range(
        db,
        db_session,
        [normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp3,
    )


def assert_time_range(
    db: DBInterface, db_session: Session, expected_run_uids: list, **filters
):
    runs = db.list_runs(db_session, **filters)
    assert len(runs) == len(expected_run_uids)
    for run in runs:
        assert run["metadata"]["uid"] in expected_run_uids
