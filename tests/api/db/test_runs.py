from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

import mlrun.api.db.sqldb.helpers
import mlrun.api.initial_data
from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.models import Run
from tests.api.db.conftest import dbs


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_runs_name_filter(db: DBInterface, db_session: Session):
    project = "project"
    run_name_1 = "run_name_1"
    run_name_2 = "run_name_2"
    run_1 = {"metadata": {"name": run_name_1}, "status": {"bla": "blabla"}}
    run_2 = {"metadata": {"name": run_name_2}, "status": {"bla": "blabla"}}
    # run with no name
    run_3 = {"metadata": {}, "status": {"bla": "blabla"}}
    run_uid_1 = "run_uid_1"
    run_uid_2 = "run_uid_2"
    run_uid_3 = "run_uid_3"

    db.store_run(db_session, run_1, run_uid_1, project)
    db.store_run(db_session, run_2, run_uid_2, project)
    db.store_run(db_session, run_3, run_uid_3, project)
    runs = db.list_runs(db_session, project=project)
    assert len(runs) == 3

    runs = db.list_runs(db_session, name=run_name_1, project=project)
    assert len(runs) == 1
    assert runs[0]["metadata"]["name"] == run_name_1

    runs = db.list_runs(db_session, name=run_name_2, project=project)
    assert len(runs) == 1
    assert runs[0]["metadata"]["name"] == run_name_2

    runs = db.list_runs(db_session, name="~RUN_naMe", project=project)
    assert len(runs) == 2


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_runs_state_filter(db: DBInterface, db_session: Session):
    project = "project-name"
    run_without_state_uid = "run_without_state_uid"
    run_without_state = {"metadata": {"uid": run_without_state_uid}, "bla": "blabla"}
    db.store_run(db_session, run_without_state, run_without_state_uid, project)

    run_with_json_state_state = "some_json_state"
    run_with_json_state_uid = "run_with_json_state_uid"
    run_with_json_state = {
        "metadata": {"uid": run_with_json_state_uid},
        "status": {"state": run_with_json_state_state},
    }
    run = Run(
        uid=run_with_json_state_uid,
        project=project,
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
        project=project,
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
        project,
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
        project=project,
        iteration=0,
        state=run_with_unequal_json_and_record_state_record_state,
        start_time=datetime.now(timezone.utc),
    )
    run.struct = run_with_unequal_json_and_record_state
    db._upsert(db_session, run, ignore=True)

    runs = db.list_runs(db_session, project=project)
    assert len(runs) == 5

    runs = db.list_runs(db_session, state=run_with_json_state_state, project=project)
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_json_state_uid

    runs = db.list_runs(db_session, state=run_with_record_state_state, project=project)
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_record_state_uid

    runs = db.list_runs(
        db_session, state=run_with_equal_json_and_record_state_state, project=project
    )
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_equal_json_and_record_state_uid

    runs = db.list_runs(
        db_session,
        state=run_with_unequal_json_and_record_state_json_state,
        project=project,
    )
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_with_unequal_json_and_record_state_uid

    runs = db.list_runs(
        db_session,
        state=run_with_unequal_json_and_record_state_record_state,
        project=project,
    )
    assert len(runs) == 0


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_run_overriding_start_time(db: DBInterface, db_session: Session):
    project = "project"
    run_name = "run_name_1"
    run = {"metadata": {"name": run_name}}
    run_uid = "run_uid"

    # First store - fills the start_time
    db.store_run(db_session, run, run_uid, project)

    # use to internal function to get the record itself to be able to assert the column itself
    runs = db._find_runs(db_session, uid=None, project=project, labels=None).all()
    assert len(runs) == 1
    assert (
        db._add_utc_timezone(runs[0].start_time).isoformat()
        == runs[0].struct["status"]["start_time"]
    )

    # Second store - should allow to override the start time
    run["status"]["start_time"] = datetime.now(timezone.utc).isoformat()
    db.store_run(db_session, run, run_uid, project)

    # get the start time and verify
    runs = db._find_runs(db_session, uid=None, project=project, labels=None).all()
    assert len(runs) == 1
    assert (
        db._add_utc_timezone(runs[0].start_time).isoformat()
        == runs[0].struct["status"]["start_time"]
    )
    assert runs[0].struct["status"]["start_time"] == run["status"]["start_time"]


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_data_migration_align_runs_table(db: DBInterface, db_session: Session):
    time_before_creation = datetime.now(tz=timezone.utc)
    # Create runs
    for project in ["run-project-1", "run-project-2", "run-project-3"]:
        for name in ["run-name-1", "run-name-2", "run-name-3"]:
            for uid in ["run-uid-1", "run-uid-2", "run-uid-3"]:
                for iter in range(3):
                    run = {
                        "metadata": {
                            "name": name,
                            "uid": uid,
                            "project": project,
                            "iter": iter,
                        }
                    }
                    db.store_run(db_session, run, uid, project, iter)
    # get all run records, change only the start_time column (and not the field in the body) to be earlier (like runs
    # will be in the field)
    runs = db._find_runs(db_session, None, "*", None).all()
    for run in runs:
        run_dict = run.struct
        assert (
            mlrun.api.db.sqldb.helpers.run_start_time(run_dict) > time_before_creation
        )
        run.start_time = time_before_creation
        db._upsert(db_session, run, ignore=True)

    mlrun.api.initial_data._align_runs_table(db, db_session)

    # assert after migration column start time aligned to the body start time
    runs = db._find_runs(db_session, None, "*", None).all()
    for run in runs:
        run_dict = run.struct
        assert mlrun.api.db.sqldb.helpers.run_start_time(
            run_dict
        ) == db._add_utc_timezone(run.start_time)
        assert (
            mlrun.api.db.sqldb.helpers.run_start_time(run_dict) > time_before_creation
        )
