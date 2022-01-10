from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

import mlrun.api.db.sqldb.helpers
import mlrun.api.initial_data
from mlrun.api.db.base import DBInterface
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
    # run with no name (had name but filled with no-name after version 2 data migration)
    run_3 = {"metadata": {"name": "no-name"}, "status": {"bla": "blabla"}}
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
    project = "project"
    run_uid_running = "run-running"
    run_uid_completed = "run-completed"
    _create_new_run(
        db,
        db_session,
        project,
        uid=run_uid_running,
        state=mlrun.runtimes.constants.RunStates.running,
    )
    _create_new_run(
        db,
        db_session,
        project,
        uid=run_uid_completed,
        state=mlrun.runtimes.constants.RunStates.completed,
    )
    runs = db.list_runs(db_session, project=project)
    assert len(runs) == 2

    runs = db.list_runs(
        db_session, project=project, state=mlrun.runtimes.constants.RunStates.running
    )
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_uid_running

    runs = db.list_runs(
        db_session, project=project, state=mlrun.runtimes.constants.RunStates.completed
    )
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_uid_completed


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_run_overriding_start_time(db: DBInterface, db_session: Session):
    # First store - fills the start_time
    project, name, uid, iteration, run = _create_new_run(db, db_session)

    # use to internal function to get the record itself to be able to assert the column itself
    runs = db._find_runs(db_session, uid=None, project=project, labels=None).all()
    assert len(runs) == 1
    assert (
        db._add_utc_timezone(runs[0].start_time).isoformat()
        == runs[0].struct["status"]["start_time"]
    )

    # Second store - should allow to override the start time
    run["status"]["start_time"] = datetime.now(timezone.utc).isoformat()
    db.store_run(db_session, run, uid, project)

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
                for iteration in range(3):
                    _create_new_run(db, db_session, project, name, uid, iteration)
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_run_success(db: DBInterface, db_session: Session):
    project, name, uid, iteration, run = _create_new_run(db, db_session)

    # use to internal function to get the record itself to be able to assert columns
    runs = db._find_runs(db_session, uid=None, project=project, labels=None).all()
    assert len(runs) == 1
    run = runs[0]
    assert run.project == project
    assert run.name == name
    assert run.uid == uid
    assert run.iteration == iteration
    assert run.state == mlrun.runtimes.constants.RunStates.created
    assert run.state == run.struct["status"]["state"]
    assert (
        db._add_utc_timezone(run.start_time).isoformat()
        == run.struct["status"]["start_time"]
    )

    assert (
        db._add_utc_timezone(run.updated).isoformat()
        == run.struct["status"]["last_update"]
    )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_update_run_success(db: DBInterface, db_session: Session):
    project, name, uid, iteration, run = _create_new_run(db, db_session)

    db.update_run(
        db_session,
        {"metadata.some-new-field": "value", "spec.another-new-field": "value"},
        uid,
        project,
        iteration,
    )
    run = db.read_run(db_session, uid, project, iteration)
    assert run["metadata"]["project"] == project
    assert run["metadata"]["name"] == name
    assert run["metadata"]["some-new-field"] == "value"
    assert run["spec"]["another-new-field"] == "value"


def _create_new_run(
    db: DBInterface,
    db_session: Session,
    project="project",
    name="run-name-1",
    uid="run-uid",
    iteration=0,
    state=mlrun.runtimes.constants.RunStates.created,
):
    run = {
        "metadata": {"name": name, "uid": uid, "project": project, "iter": iteration},
        "status": {"state": state},
    }

    db.store_run(db_session, run, uid, project, iter=iteration)
    return project, name, uid, iteration, run
