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
def test_storing_project_background_task(db: DBInterface, db_session: Session):
    task_name = "test"
    project = "test-project"
    project_timeout = 40
    no_project_timeout = 1
    db.store_background_task(
        db_session, "test", timeout=project_timeout, project=project
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
