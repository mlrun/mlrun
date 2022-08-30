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
def test_store_project_background_task(db: DBInterface, db_session: Session):
    project = "test-project"
    db.store_background_task(db_session, "test", timeout=600, project=project)
    background_task = db.get_background_task(db_session, "test", project=project)
    assert background_task.metadata.name == "test"
    assert background_task.status.state == "running"


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_get_project_background_task_with_timeout_exceeded(
    db: DBInterface, db_session: Session
):
    project = "test-project"
    db.store_background_task(db_session, "test", timeout=1, project=project)
    background_task = db.get_background_task(db_session, "test", project=project)
    assert background_task.status.state == "running"
    time.sleep(1)
    background_task = db.get_background_task(db_session, "test", project=project)
    assert background_task.status.state == "failed"


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_get_project_background_task_doesnt_exists(
    db: DBInterface, db_session: Session
):
    project = "test-project"
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_background_task(db_session, "test", project=project)


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_project_background_task_after_status_updated(
    db: DBInterface, db_session: Session
):
    project = "test-project"
    db.store_background_task(db_session, "test", project=project)
    background_task = db.get_background_task(db_session, "test", project=project)
    assert background_task.status.state == schemas.BackgroundTaskState.running

    db.store_background_task(
        db_session, "test", state=schemas.BackgroundTaskState.failed, project=project
    )
    background_task = db.get_background_task(db_session, "test", project=project)
    assert background_task.status.state == schemas.BackgroundTaskState.failed

    # Expecting to fail
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        db.store_background_task(
            db_session,
            "test",
            state=schemas.BackgroundTaskState.running,
            project=project,
        )
    # expecting to fail, because terminal state is terminal which means it is not supposed to change
    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        db.store_background_task(
            db_session,
            "test",
            state=schemas.BackgroundTaskState.succeeded,
            project=project,
        )

    db.store_background_task(
        db_session, "test", state=schemas.BackgroundTaskState.failed, project=project
    )


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_get_project_background_task_with_disabled_timeout(
    db: DBInterface, db_session: Session
):
    task_name = "test"
    project = "test-project"
    task_timeout = 0
    mlrun.mlconf.background_tasks.timeout_mode = "disabled"
    db.store_background_task(
        db_session, name=task_name, timeout=task_timeout, project=project
    )
    background_task = db.get_background_task(db_session, task_name, project)
    # expecting to be None because if mode is disabled and timeout provided it ignores it
    assert background_task.metadata.timeout is None
    # expecting created and updated time to be equal because mode disabled even if timeout exceeded
    assert background_task.metadata.created == background_task.metadata.updated
    assert background_task.status.state == mlrun.api.schemas.BackgroundTaskState.running
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
