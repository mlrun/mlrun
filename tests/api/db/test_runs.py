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
def test_list_distinct_runs_uids(db: DBInterface, db_session: Session):
    project_name = "project"
    # create 3 runs with same uid but different iterations
    for i in range(3):
        _create_new_run(db, db_session, project=project_name, iteration=i)

    runs = db.list_runs(db_session, project=project_name, iter=True)
    assert len(runs) == 3

    distinct_runs = db.list_distinct_runs_uids(
        db_session, project=project_name, only_uids=False
    )
    assert len(distinct_runs) == 1
    assert type(distinct_runs[0]) == dict

    only_uids = db.list_distinct_runs_uids(
        db_session, project=project_name, only_uids=True
    )
    assert len(only_uids) == 1
    assert type(only_uids[0]) == str

    only_uids_requested_true = db.list_distinct_runs_uids(
        db_session, project=project_name, only_uids=True, requested_logs=True
    )
    assert len(only_uids_requested_true) == 0

    only_uids_requested_false = db.list_distinct_runs_uids(
        db_session, project=project_name, only_uids=True, requested_logs=False
    )
    assert len(only_uids_requested_false) == 1
    assert type(only_uids[0]) == str

    distinct_runs_requested_true = db.list_distinct_runs_uids(
        db_session, project=project_name, requested_logs=True
    )
    assert len(distinct_runs_requested_true) == 0

    distinct_runs_requested_false = db.list_distinct_runs_uids(
        db_session, project=project_name, requested_logs=False
    )
    assert len(distinct_runs_requested_false) == 1
    assert type(distinct_runs[0]) == dict


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
        db_session, project=project, states=[mlrun.runtimes.constants.RunStates.running]
    )
    assert len(runs) == 1
    assert runs[0]["metadata"]["uid"] == run_uid_running

    runs = db.list_runs(
        db_session,
        project=project,
        states=[mlrun.runtimes.constants.RunStates.completed],
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
            for index in range(3):
                uid = f"{name}-uid-{index}"
                for iteration in range(3):
                    _create_new_run(
                        db,
                        db_session,
                        project,
                        name,
                        uid,
                        iteration,
                        state=mlrun.runtimes.constants.RunStates.completed,
                    )
    # get all run records, and change to be as they will be in field (before the migration)
    runs = db._find_runs(db_session, None, "*", None).all()
    for run in runs:
        _change_run_record_to_before_align_runs_migration(run, time_before_creation)
        db._upsert(db_session, [run], ignore=True)

    # run the migration
    mlrun.api.initial_data._align_runs_table(db, db_session)

    # assert after migration column start time aligned to the body start time
    runs = db._find_runs(db_session, None, "*", None).all()
    for run in runs:
        _ensure_run_after_align_runs_migration(db, run, time_before_creation)


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_data_migration_align_runs_table_with_empty_run_body(
    db: DBInterface, db_session: Session
):
    time_before_creation = datetime.now(tz=timezone.utc)
    # First store - fills the start_time
    project, name, uid, iteration, run = _create_new_run(
        db, db_session, state=mlrun.runtimes.constants.RunStates.completed
    )
    # get all run records, and change to be as they will be in field (before the migration)
    runs = db._find_runs(db_session, None, "*", None).all()
    assert len(runs) == 1
    run = runs[0]
    # change to be as it will be in field (before the migration) and then empty the body
    _change_run_record_to_before_align_runs_migration(run, time_before_creation)
    run.struct = {}
    db._upsert(db_session, [run], ignore=True)

    # run the migration
    mlrun.api.initial_data._align_runs_table(db, db_session)

    runs = db._find_runs(db_session, None, "*", None).all()
    assert len(runs) == 1
    run = runs[0]
    _ensure_run_after_align_runs_migration(db, run)


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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_and_update_run_update_name_failure(db: DBInterface, db_session: Session):
    project, name, uid, iteration, run = _create_new_run(db, db_session)

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="Changing name for an existing run is invalid",
    ):
        run["metadata"]["name"] = "new-name"
        db.store_run(
            db_session,
            run,
            uid,
            project,
            iteration,
        )

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="Changing name for an existing run is invalid",
    ):
        db.update_run(
            db_session,
            {"metadata.name": "new-name"},
            uid,
            project,
            iteration,
        )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_runs_limited_unsorted_failure(db: DBInterface, db_session: Session):
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="Limiting the number of returned records without sorting will provide non-deterministic results",
    ):
        db.list_runs(
            db_session,
            sort=False,
            last=1,
        )


def _change_run_record_to_before_align_runs_migration(run, time_before_creation):
    run_dict = run.struct

    # change only the start_time column (and not the field in the body) to be earlier
    assert mlrun.api.db.sqldb.helpers.run_start_time(run_dict) > time_before_creation
    run.start_time = time_before_creation

    # change name column to be empty
    run.name = None

    # change state column to be empty created (should be completed)
    run.state = mlrun.runtimes.constants.RunStates.created

    # change updated column to be empty
    run.updated = None


def _ensure_run_after_align_runs_migration(
    db: DBInterface, run, time_before_creation=None
):
    run_dict = run.struct

    # ensure start time aligned
    assert mlrun.api.db.sqldb.helpers.run_start_time(run_dict) == db._add_utc_timezone(
        run.start_time
    )
    if time_before_creation is not None:
        assert (
            mlrun.api.db.sqldb.helpers.run_start_time(run_dict) > time_before_creation
        )

    # ensure name column filled
    assert run_dict["metadata"]["name"] == run.name

    # ensure state column aligned
    assert run_dict["status"]["state"] == run.state

    # ensure updated column filled
    assert (
        run_dict["status"]["last_update"]
        == db._add_utc_timezone(run.updated).isoformat()
    )


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
