# Copyright 2026 Iguazio
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

import unittest.mock
import uuid

import pytest

import mlrun.errors

import framework.api.utils
import framework.utils.background_tasks


class TestRunProject2pcRunnerInline:
    async def test_awaits_runner_with_provided_session(self):
        runner = unittest.mock.AsyncMock()
        await framework.api.utils.run_project_2pc_runner_inline(
            runner, "p1", db_session="sess"
        )
        runner.assert_awaited_once_with("sess")

    async def test_swallows_exception_group(self):
        # 2PC partial failures arrive as ExceptionGroup. The HTTP request
        # must still succeed; reconciliation will retry the row.
        async def runner(_session):
            raise ExceptionGroup("follower failures", [RuntimeError("a")])

        # If swallowed correctly this returns None without raising.
        await framework.api.utils.run_project_2pc_runner_inline(
            runner, "p1", db_session="sess"
        )

    async def test_propagates_non_exception_group(self):
        # A leader-side primitive failure (e.g. precondition mismatch from a
        # DB advance/complete call) is not an ExceptionGroup and signals a
        # broken invariant — surface it to the caller.
        async def runner(_session):
            raise mlrun.errors.MLRunPreconditionFailedError("bad state")

        with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
            await framework.api.utils.run_project_2pc_runner_inline(
                runner, "p1", db_session="sess"
            )


class TestGetOrCreateProject2pcBackgroundTask:
    @pytest.fixture
    def mock_handler_cls(self):
        # Replace the handler class with a callable that returns our mock —
        # both the existing-task lookup and the create call go through this
        # one object, so we can drive both branches deterministically.
        handler = unittest.mock.MagicMock()
        with unittest.mock.patch.object(
            framework.utils.background_tasks,
            "InternalBackgroundTasksHandler",
            return_value=handler,
        ):
            yield handler

    def test_returns_existing_task_when_active_kind_exists(self, mock_handler_cls):
        # Coalescing: a second request for the same project must NOT spawn a
        # parallel orchestration; it must reuse the existing task name so
        # callers can poll the same record.
        runner = unittest.mock.Mock()
        existing_task = unittest.mock.Mock()
        existing_task.metadata.name = "existing-task-name"
        mock_handler_cls.get_active_background_task_by_kind.return_value = existing_task

        task, name = framework.api.utils.get_or_create_project_2pc_background_task(
            "p1", uuid.uuid4(), runner
        )

        assert task is None
        assert name == "existing-task-name"
        # Must short-circuit before allocating a new task.
        mock_handler_cls.create_background_task.assert_not_called()

    def test_creates_new_task_when_no_active_kind(self, mock_handler_cls):
        runner = unittest.mock.Mock()
        mock_handler_cls.get_active_background_task_by_kind.side_effect = (
            mlrun.errors.MLRunNotFoundError("no active task")
        )
        sentinel_callable = unittest.mock.Mock()
        mock_handler_cls.create_background_task.return_value = (
            sentinel_callable,
            "new-task-name",
        )

        task, name = framework.api.utils.get_or_create_project_2pc_background_task(
            "p1", uuid.uuid4(), runner
        )

        assert task is sentinel_callable
        assert name == "new-task-name"
        # Kind is per-project so two different projects coexist.
        kind_arg = mock_handler_cls.create_background_task.call_args.args[0]
        assert kind_arg == "project.sync.2pc.p1"

    def test_kind_is_per_project(self, mock_handler_cls):
        # The lock is per-project so concurrent requests against different
        # projects don't collide. Verify by invoking twice with different
        # names and checking the kind argument we hand to the handler.
        mock_handler_cls.get_active_background_task_by_kind.side_effect = (
            mlrun.errors.MLRunNotFoundError("no active task")
        )
        mock_handler_cls.create_background_task.return_value = (
            unittest.mock.Mock(),
            "n",
        )

        framework.api.utils.get_or_create_project_2pc_background_task(
            "alpha", uuid.uuid4(), unittest.mock.Mock()
        )
        framework.api.utils.get_or_create_project_2pc_background_task(
            "beta", uuid.uuid4(), unittest.mock.Mock()
        )

        kinds = [
            call.args[0]
            for call in mock_handler_cls.create_background_task.call_args_list
        ]
        assert kinds == ["project.sync.2pc.alpha", "project.sync.2pc.beta"]
