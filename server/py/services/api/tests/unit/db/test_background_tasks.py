# Copyright 2023 Iguazio
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
import datetime
import time
import unittest.mock

import pytest

import mlrun.common.schemas
import mlrun.errors

from framework.tests.unit.db.common_fixtures import TestDatabaseBase
from framework.utils.background_tasks import background_task_exceeded_timeout


class TestBackgroundTasks(TestDatabaseBase):
    def test_store_project_background_task(self):
        project = "test-project"
        self._db.store_background_task(
            self._db_session, "test", timeout=600, project=project
        )
        background_task = self._db.get_background_task(
            self._db_session,
            "test",
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert background_task.metadata.name == "test"
        assert background_task.status.state == "running"

    def test_get_project_background_task_with_timeout_exceeded(self):
        project = "test-project"
        self._db.store_background_task(
            self._db_session, "test", timeout=1, project=project
        )
        background_task = self._db.get_background_task(
            self._db_session,
            "test",
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert background_task.status.state == "running"
        time.sleep(1)
        background_task = self._db.get_background_task(
            self._db_session,
            "test",
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert background_task.status.state == "failed"

    def test_get_project_background_task_doesnt_exists(self):
        project = "test-project"
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.get_background_task(
                self._db_session,
                "test",
                project=project,
                background_task_exceeded_timeout_func=background_task_exceeded_timeout,
            )

    def test_store_project_background_task_after_status_updated(self):
        project = "test-project"
        self._db.store_background_task(self._db_session, "test", project=project)
        background_task = self._db.get_background_task(
            self._db_session,
            "test",
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.running
        )

        self._db.store_background_task(
            self._db_session,
            "test",
            state=mlrun.common.schemas.BackgroundTaskState.failed,
            project=project,
        )
        background_task = self._db.get_background_task(
            self._db_session,
            "test",
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.failed
        )

        # Expecting to fail
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            self._db.store_background_task(
                self._db_session,
                "test",
                state=mlrun.common.schemas.BackgroundTaskState.running,
                project=project,
            )
        # expecting to fail, because terminal state is terminal which means it is not supposed to change
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            self._db.store_background_task(
                self._db_session,
                "test",
                state=mlrun.common.schemas.BackgroundTaskState.succeeded,
                project=project,
            )

        self._db.store_background_task(
            self._db_session,
            "test",
            state=mlrun.common.schemas.BackgroundTaskState.failed,
            project=project,
        )

    def test_get_project_background_task_with_disabled_timeout(self):
        task_name = "test"
        project = "test-project"
        task_timeout = 0
        mlrun.mlconf.background_tasks.timeout_mode = "disabled"
        self._db.store_background_task(
            self._db_session, name=task_name, timeout=task_timeout, project=project
        )
        background_task = self._db.get_background_task(
            self._db_session,
            task_name,
            project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        # expecting to be None because if mode is disabled and timeout provided it ignores it
        assert background_task.metadata.timeout is None
        # expecting created and updated time to be equal because mode disabled even if timeout exceeded
        assert background_task.metadata.created == background_task.metadata.updated
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.running
        )
        task_name = "test1"
        self._db.store_background_task(
            self._db_session, name=task_name, project=project
        )
        # because timeout default mode is disabled, expecting not to enrich the background task timeout
        background_task = self._db.get_background_task(
            self._db_session,
            task_name,
            project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert background_task.metadata.timeout is None
        assert background_task.metadata.created == background_task.metadata.updated
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.running
        )

        self._db.store_background_task(
            self._db_session,
            name=task_name,
            project=project,
            state=mlrun.common.schemas.BackgroundTaskState.succeeded,
        )
        background_task_new = self._db.get_background_task(
            self._db_session,
            task_name,
            project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )
        assert (
            background_task_new.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )
        assert background_task_new.metadata.updated > background_task.metadata.updated
        assert background_task_new.metadata.created == background_task.metadata.created

    def test_list_project_background_task_filters(self):
        project = "test-project"
        running = "running"
        failed = "failed"
        succeeded = "succeeded"
        old_task = "old_task"

        self._db.store_background_task(
            self._db_session, running, timeout=600, project=project
        )
        self._db.store_background_task(
            self._db_session,
            failed,
            timeout=600,
            project=project,
            state=mlrun.common.schemas.BackgroundTaskState.failed,
        )
        self._db.store_background_task(
            self._db_session,
            succeeded,
            timeout=600,
            project=project,
            state=mlrun.common.schemas.BackgroundTaskState.succeeded,
        )

        with unittest.mock.patch(
            "mlrun.utils.now_date",
            return_value=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=10),
        ):
            self._db.store_background_task(
                self._db_session, old_task, timeout=600, project=project
            )

        background_tasks = self._db.list_background_tasks(
            self._db_session,
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
        )

        assert len(background_tasks) == 4
        background_task_names = [task.metadata.name for task in background_tasks]
        for name in [running, failed, succeeded, old_task]:
            assert name in background_task_names

        # test created_from filter
        background_tasks = self._db.list_background_tasks(
            self._db_session,
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
            created_from=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=5),
        )

        assert len(background_tasks) == 3
        background_task_names = [task.metadata.name for task in background_tasks]
        for name in [running, failed, succeeded]:
            assert name in background_task_names

        # test last_update_time_from filters
        background_tasks = self._db.list_background_tasks(
            self._db_session,
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
            last_update_time_from=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=5),
        )

        assert len(background_tasks) == 3
        background_task_names = [task.metadata.name for task in background_tasks]
        for name in [running, failed, succeeded]:
            assert name in background_task_names

        # test last_update_time_to filters
        background_tasks = self._db.list_background_tasks(
            self._db_session,
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
            last_update_time_to=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=5),
        )

        assert len(background_tasks) == 1
        background_task_names = [task.metadata.name for task in background_tasks]
        for name in [old_task]:
            assert name in background_task_names

        # test state filter
        background_tasks = self._db.list_background_tasks(
            self._db_session,
            project=project,
            background_task_exceeded_timeout_func=background_task_exceeded_timeout,
            states=[
                mlrun.common.schemas.BackgroundTaskState.failed,
                mlrun.common.schemas.BackgroundTaskState.succeeded,
            ],
        )

        assert len(background_tasks) == 2
        background_task_names = [task.metadata.name for task in background_tasks]
        for name in [failed, succeeded]:
            assert name in background_task_names
