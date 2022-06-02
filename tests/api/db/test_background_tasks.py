import datetime
import time

import pytest
from sqlalchemy.orm import Session

import mlrun.api.initial_data
import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_background_task(db: DBInterface, db_session: Session):
    db.store_background_task(db_session, "test", timeout=600)
    background_task = db.get_background_task(db_session, "test")
    assert background_task.metadata.name == "test"
    assert background_task.status.state == "running"


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_get_background_task_with_timeout_exceeded(
    db: DBInterface, db_session: Session
):
    db.store_background_task(db_session, "test", timeout=1)
    background_task = db.get_background_task(db_session, "test")
    assert background_task.status.state == "running"
    time.sleep(1)
    background_task = db.get_background_task(db_session, "test")
    assert background_task.status.state == "failed"


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_get_background_task_doesnt_exists(db: DBInterface, db_session: Session):
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_background_task(db_session, "test")


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_background_task_after_status_updated(
    db: DBInterface, db_session: Session
):
    db.store_background_task(db_session, "test")
    background_task = db.get_background_task(db_session, "test")
    assert background_task.status.state == schemas.BackgroundTaskState.running

    db.store_background_task(
        db_session, "test", state=schemas.BackgroundTaskState.failed
    )
    background_task = db.get_background_task(db_session, "test")
    assert background_task.status.state == schemas.BackgroundTaskState.failed

    # Expecting to fail
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        db.store_background_task(
            db_session, "test", state=schemas.BackgroundTaskState.running
        )
    # expecting to fail, because terminal state is terminal which means it is not supposed to change
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        db.store_background_task(
            db_session, "test", state=schemas.BackgroundTaskState.succeeded
        )

    db.store_background_task(
        db_session, "test", state=schemas.BackgroundTaskState.failed
    )


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_storing_project_background_task_and_non_project_background_task_with_same_name(
    db: DBInterface, db_session: Session
):
    task_name = "test"
    project = "test-project"
    project_timeout = 40
    no_project_timeout = 1
    db.store_background_task(
        db_session, name=task_name, timeout=project_timeout, project=project
    )
    # expecting to fail because there is no background task named tested without a project
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_background_task(db_session, task_name)
    background_task = db.get_background_task(db_session, task_name, project=project)
    assert background_task.metadata.project == project
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running

    db.store_background_task(db_session, task_name, timeout=no_project_timeout)
    background_task = db.get_background_task(db_session, task_name)
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    assert background_task.metadata.timeout == no_project_timeout
    assert background_task.metadata.project is None
    time.sleep(no_project_timeout)

    background_task = db.get_background_task(db_session, task_name)
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.failed
    assert background_task.metadata.timeout == no_project_timeout
    assert background_task.metadata.project is None

    db.store_background_task(
        db_session,
        task_name,
        state=mlrun.api.schemas.BackgroundTaskState.succeeded,
        project=project,
    )
    background_task = db.get_background_task(db_session, task_name, project=project)
    assert (
        background_task.status.state == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.metadata.timeout == project_timeout
    assert background_task.metadata.project == project


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_get_background_task_with_exceeded_timeout_and_disabled_timeout(
    db: DBInterface, db_session: Session
):
    task_name = "test"
    project = "test-project"
    task_timeout = 0
    mlrun.mlconf.background_tasks_timeout_defaults.mode = "disabled"
    db.store_background_task(
        db_session, name=task_name, timeout=task_timeout, project=project
    )
    background_task = db.get_background_task(db_session, task_name, project)
    assert background_task.metadata.timeout == task_timeout
    # expecting created and updated time to be equal because mode disabled even if timeout exceeded
    assert background_task.metadata.created == background_task.metadata.updated
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
    # expecting timeout to exceed
    assert (
        datetime.timedelta(seconds=background_task.metadata.timeout)
        + background_task.metadata.updated
        < datetime.datetime.utcnow()
    )
    task_name = "test1"
    db.store_background_task(db_session, name=task_name, project=project)
    # because timeout default mode is disabled, expecting not to enrich the background task timeout
    background_task = db.get_background_task(db_session, task_name, project)
    assert background_task.metadata.timeout is None
    assert background_task.metadata.created == background_task.metadata.updated
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running

    db.store_background_task(
        db_session,
        name=task_name,
        project=project,
        state=mlrun.api.schemas.BackgroundTaskState.succeeded,
    )
    background_task_new = db.get_background_task(db_session, task_name, project)
    assert (
        background_task_new.status.state
        == mlrun.api.schemas.BackgroundTaskState.succeeded
    )
    assert background_task_new.metadata.updated > background_task.metadata.updated
    assert background_task_new.metadata.created == background_task.metadata.created
