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

import asyncio
import functools
import inspect
import unittest.mock
import uuid

import pytest

import mlrun.common.schemas
import mlrun.config
import mlrun.errors

import framework.api.utils
import framework.utils.projects.leader
import services.api.crud


def _make_member(
    *,
    leader_follower,
    followers: dict[str, object],
) -> framework.utils.projects.leader.Member:
    """
    Build a Member without going through __init__/initialize so tests can
    inject mocks for the leader-follower and follower fan-out targets. The
    real Member is a singleton — bypassing __new__ on the metaclass dodges
    that and gives every test a fresh instance.
    """
    member = framework.utils.projects.leader.Member.__new__(
        framework.utils.projects.leader.Member
    )
    member._leader_follower = leader_follower
    member._followers = followers
    member._projects_in_deletion = set()
    member._inflight_retries = set()
    return member


def _make_stale_project(
    name: str,
    state: mlrun.common.schemas.ProjectState,
    *,
    phase: int = 0,
    op_id: uuid.UUID | None = None,
) -> mlrun.common.schemas.ProjectOut:
    return mlrun.common.schemas.ProjectOut(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name),
        status=mlrun.common.schemas.ProjectStatus(
            state=state,
            op_id=op_id or uuid.uuid4(),
            phase=phase,
        ),
    )


def _make_project(name: str = "p1") -> mlrun.common.schemas.Project:
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name),
        spec=mlrun.common.schemas.ProjectSpec(description="d"),
    )


class TestCrudLeaderFollowerAccessor:
    def test_returns_leader_when_it_is_crud_projects(self):
        crud = services.api.crud.Projects()
        member = _make_member(leader_follower=crud, followers={})
        assert member._crud_leader_follower is crud

    def test_raises_when_leader_is_not_crud_projects(self):
        # Any non-crud follower stand-in trips the runtime safety check; the
        # error message must call out the actual type to aid debugging.
        not_crud = unittest.mock.Mock(spec=[])
        member = _make_member(leader_follower=not_crud, followers={})
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            _ = member._crud_leader_follower
        assert "2PC project sync requires" in str(exc_info.value)


class TestRunOnAllFollowersInParallel:
    async def test_calls_each_follower_with_args(self):
        f1 = unittest.mock.Mock()
        f2 = unittest.mock.Mock()
        member = _make_member(
            leader_follower=services.api.crud.Projects(),
            followers={"a": f1, "b": f2},
        )

        await member._run_on_all_followers_in_parallel(
            "prepare_create_project", "proj", "op"
        )

        f1.prepare_create_project.assert_called_once_with("proj", "op")
        f2.prepare_create_project.assert_called_once_with("proj", "op")

    async def test_aggregates_failures_into_exception_group(self):
        # Both followers must run to completion — we should not see a
        # short-circuit after the first error — and both errors must show up
        # in the resulting ExceptionGroup so reconciliation gets the full
        # picture.
        f1 = unittest.mock.Mock()
        f1.commit_create_project.side_effect = RuntimeError("a-failed")
        f2 = unittest.mock.Mock()
        f2.commit_create_project.side_effect = ValueError("b-failed")

        member = _make_member(
            leader_follower=services.api.crud.Projects(),
            followers={"a": f1, "b": f2},
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            await member._run_on_all_followers_in_parallel(
                "commit_create_project", "proj", "op"
            )

        f1.commit_create_project.assert_called_once()
        f2.commit_create_project.assert_called_once()
        # Both underlying exceptions carried, types preserved.
        types_seen = {type(e) for e in exc_info.value.exceptions}
        assert types_seen == {RuntimeError, ValueError}

    async def test_no_followers_is_noop(self):
        member = _make_member(
            leader_follower=services.api.crud.Projects(), followers={}
        )
        # Must not raise, must not call into anything: the leader handles
        # its own row in the surrounding orchestrator.
        await member._run_on_all_followers_in_parallel("prepare_create_project")


@pytest.fixture
def crud_mock() -> unittest.mock.MagicMock:
    """A mock that satisfies the isinstance check in _crud_leader_follower."""
    return unittest.mock.MagicMock(spec=services.api.crud.Projects)


class TestRunCreateFlow:
    async def test_phase_zero_runs_full_pipeline(self, crud_mock):
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        await member._run_create_flow(project, op_id, db_session="sess")

        # Full ordered pipeline: prepare → advance → commit → complete.
        follower.prepare_create_project.assert_called_once_with(project, op_id)
        crud_mock.advance_create_project_to_commit.assert_called_once_with(
            "sess", "p1", op_id
        )
        follower.commit_create_project.assert_called_once_with("p1", op_id)
        crud_mock.complete_create_project.assert_called_once_with("sess", "p1", op_id)

    async def test_phase_one_resumes_only_commit(self, crud_mock):
        # Resume-from-crash path: row is at phase=1 because a previous run
        # finished `advance` but never reached `complete`.
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 1
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        await member._run_create_flow(project, op_id, db_session="sess")

        follower.prepare_create_project.assert_not_called()
        crud_mock.advance_create_project_to_commit.assert_not_called()
        follower.commit_create_project.assert_called_once_with("p1", op_id)
        crud_mock.complete_create_project.assert_called_once()

    async def test_phase_none_means_superseded_skip_everything(self, crud_mock):
        # phase=None signals a newer op_id has taken over; this orchestration
        # must do nothing rather than racing the new owner.
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = None
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        await member._run_create_flow(project, op_id, db_session="sess")

        follower.prepare_create_project.assert_not_called()
        follower.commit_create_project.assert_not_called()
        crud_mock.advance_create_project_to_commit.assert_not_called()
        crud_mock.complete_create_project.assert_not_called()

    async def test_prepare_failure_does_not_advance(self, crud_mock):
        # If prepare fails, we MUST NOT advance the row to phase=1 — that
        # would falsely claim every follower is ready to commit.
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()
        follower.prepare_create_project.side_effect = RuntimeError("nope")

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        with pytest.raises(ExceptionGroup):
            await member._run_create_flow(project, op_id, db_session="sess")

        crud_mock.advance_create_project_to_commit.assert_not_called()
        crud_mock.complete_create_project.assert_not_called()

    async def test_commit_failure_does_not_complete(self, crud_mock):
        # Same invariant on the commit step: a failed commit must leave the
        # row at phase=1 so reconciliation can retry the commit cleanly.
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 1
        follower = unittest.mock.Mock()
        follower.commit_create_project.side_effect = RuntimeError("nope")

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        with pytest.raises(ExceptionGroup):
            await member._run_create_flow(project, op_id, db_session="sess")

        crud_mock.complete_create_project.assert_not_called()


class TestRunUpdateFlow:
    async def test_phase_zero_fans_out_then_completes(self, crud_mock):
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        await member._run_update_flow("p1", project, op_id, db_session="sess")

        follower.update_project_follower.assert_called_once_with("p1", project, op_id)
        crud_mock.complete_update_project.assert_called_once_with("sess", "p1", op_id)

    async def test_phase_none_skips_everything(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = None
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        await member._run_update_flow("p1", _make_project(), op_id, db_session="sess")

        follower.update_project_follower.assert_not_called()
        crud_mock.complete_update_project.assert_not_called()

    async def test_follower_failure_blocks_complete(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()
        follower.update_project_follower.side_effect = RuntimeError("boom")

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})

        with pytest.raises(ExceptionGroup):
            await member._run_update_flow(
                "p1", _make_project(), op_id, db_session="sess"
            )

        crud_mock.complete_update_project.assert_not_called()


class TestRunDeleteFlow:
    async def test_phase_zero_runs_full_pipeline_with_post_delete(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})
        # post_delete_project is an async hook on the base member; stub it so
        # we can assert it ran exactly once after the row was removed.
        member.post_delete_project = unittest.mock.AsyncMock()

        await member._run_delete_flow("p1", op_id, db_session="sess")

        follower.prepare_delete_project.assert_called_once_with("p1", op_id)
        crud_mock.advance_delete_project_to_commit.assert_called_once_with(
            "sess", "p1", op_id
        )
        follower.commit_delete_project.assert_called_once_with("p1", op_id)
        crud_mock.complete_delete_project.assert_called_once_with("sess", "p1", op_id)
        # post_delete_project must run only after the row is gone — i.e. only
        # if complete_delete_project was reached.
        member.post_delete_project.assert_awaited_once_with("p1")

    async def test_phase_one_skips_prepare_and_advance(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = 1
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})
        member.post_delete_project = unittest.mock.AsyncMock()

        await member._run_delete_flow("p1", op_id, db_session="sess")

        follower.prepare_delete_project.assert_not_called()
        crud_mock.advance_delete_project_to_commit.assert_not_called()
        follower.commit_delete_project.assert_called_once()
        crud_mock.complete_delete_project.assert_called_once()
        member.post_delete_project.assert_awaited_once_with("p1")

    async def test_phase_none_skips_post_delete(self, crud_mock):
        # post_delete_project drops log streams. If the row is no longer
        # ours, another op already owns the cleanup — running it here would
        # double-delete log resources for that other op.
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = None
        follower = unittest.mock.Mock()

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})
        member.post_delete_project = unittest.mock.AsyncMock()

        await member._run_delete_flow("p1", op_id, db_session="sess")

        follower.prepare_delete_project.assert_not_called()
        follower.commit_delete_project.assert_not_called()
        member.post_delete_project.assert_not_awaited()

    async def test_commit_failure_blocks_complete_and_post_delete(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = 1
        follower = unittest.mock.Mock()
        follower.commit_delete_project.side_effect = RuntimeError("nope")

        member = _make_member(leader_follower=crud_mock, followers={"nuc": follower})
        member.post_delete_project = unittest.mock.AsyncMock()

        with pytest.raises(ExceptionGroup):
            await member._run_delete_flow("p1", op_id, db_session="sess")

        crud_mock.complete_delete_project.assert_not_called()
        member.post_delete_project.assert_not_awaited()


class TestPublicLeaderEntrypoints:
    """
    The public Member methods (delete_project / store_project / etc.) return
    a 3rd item — a ``ProjectSyncRunner | None`` — only when 2PC is enabled.
    Verify routing without driving the full async flow.
    """

    @pytest.fixture(autouse=True)
    def reset_2pc_gate(self):
        original = (
            mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc
        )
        try:
            yield
        finally:
            mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
                original
            )

    def test_delete_project_returns_runner_when_2pc_enabled(self, crud_mock):
        mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
            "enabled"
        )
        op_id = uuid.uuid4()
        # Member.delete_project routes through `_crud_leader_follower.begin_delete_project`
        # which it expects to either return (op_id, updated_at) or None.
        crud_mock.begin_delete_project.return_value = (op_id, "now")
        member = _make_member(leader_follower=crud_mock, followers={})

        is_running, runner = member.delete_project("sess", "p1")

        assert is_running is False
        assert runner is not None
        # The runner is a partial bound to _run_delete_flow with name + op_id.
        assert isinstance(runner, functools.partial)
        assert runner.func == member._run_delete_flow
        assert runner.args == ("p1", op_id)

    def test_delete_project_returns_none_runner_when_begin_says_skip(self, crud_mock):
        # begin_delete_project returns None when the project is already gone
        # or the strategy was ``check`` — no orchestration to run.
        mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
            "enabled"
        )
        crud_mock.begin_delete_project.return_value = None
        member = _make_member(leader_follower=crud_mock, followers={})

        is_running, runner = member.delete_project("sess", "p1")

        assert is_running is False
        assert runner is None

    def test_delete_project_returns_none_runner_when_2pc_disabled(self):
        mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
            "disabled"
        )
        # Legacy path uses _run_on_all_followers; stub via _leader_follower
        # and an empty followers dict so the call is a no-op.
        leader_follower = unittest.mock.MagicMock(spec=services.api.crud.Projects)
        member = _make_member(leader_follower=leader_follower, followers={})

        is_running, runner = member.delete_project("sess", "p1")

        assert is_running is False
        assert runner is None
        leader_follower.delete_project.assert_called_once()

    def test_sync_projects_raises_when_2pc_enabled(self):
        # The legacy periodic sync is incompatible with the 2PC state
        # machine; if the gate is on, the loop must refuse to run rather
        # than corrupt rows.
        mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
            "enabled"
        )
        member = _make_member(
            leader_follower=services.api.crud.Projects(), followers={}
        )
        with pytest.raises(NotImplementedError):
            member._sync_projects()


class TestRetryStuckProjects:
    """
    Reconciliation loop body. Verifies dispatch-by-state without driving the
    full 2PC flow — the actual flow methods are covered by TestRunCreateFlow
    / TestRunUpdateFlow / TestRunDeleteFlow above.
    """

    @pytest.fixture
    def session_helper_mock(self, monkeypatch):
        """
        Stand-in for ``run_async_function_with_new_db_session``. Records each
        call (so tests can introspect what was wrapped) and invokes the
        target with a sentinel session — async targets are awaited, sync
        ones are called. The real helper does the same routing under
        threadpool/async-context-manager hops; for unit tests, eagerly
        running the target is enough.
        """
        calls: list[tuple] = []
        sentinel_session = unittest.mock.Mock(name="fake_session")

        async def helper(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            target = func.func if isinstance(func, functools.partial) else func
            if inspect.iscoroutinefunction(target):
                return await func(sentinel_session, *args, **kwargs)
            return func(sentinel_session, *args, **kwargs)

        monkeypatch.setattr(
            "framework.db.session.run_async_function_with_new_db_session",
            helper,
        )
        return calls

    @pytest.fixture
    def get_or_create_2pc_task_mock(self, monkeypatch):
        """
        Stand-in for ``get_or_create_project_2pc_background_task``. Records
        each ``(project_name, runner)`` call and returns a fresh AsyncMock
        callable so tests can assert dispatch + (optionally) await it.
        Override ``state["result"]`` to return ``(None, name)`` and exercise
        the "active task already exists" branch.
        """
        state = {"calls": [], "result": None}

        def factory(project_name, runner):
            state["calls"].append((project_name, runner))
            if state["result"] is not None:
                return state["result"]
            return unittest.mock.AsyncMock(), "task-name"

        monkeypatch.setattr(
            "framework.api.utils.get_or_create_project_2pc_background_task",
            factory,
        )
        return state

    @pytest.fixture
    def inline_runner_mock(self, monkeypatch):
        mock = unittest.mock.AsyncMock()
        monkeypatch.setattr(
            "framework.api.utils.run_project_2pc_runner_inline",
            mock,
        )
        return mock

    @pytest.fixture
    def crud_mock(self) -> unittest.mock.MagicMock:
        return unittest.mock.MagicMock(spec=services.api.crud.Projects)

    async def test_dispatches_creating_via_background_task(
        self,
        crud_mock,
        session_helper_mock,
        get_or_create_2pc_task_mock,
        inline_runner_mock,
    ):
        # creating → wrap in get_or_create_project_2pc_background_task with a
        # partial bound to _run_create_flow(project, op_id), and schedule the
        # returned callable.
        op_id = uuid.uuid4()
        project = _make_stale_project(
            "p1", mlrun.common.schemas.ProjectState.creating, op_id=op_id
        )
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[project])
        )
        fake_callable = unittest.mock.AsyncMock()
        get_or_create_2pc_task_mock["result"] = (fake_callable, "task-name")

        member = _make_member(leader_follower=crud_mock, followers={})
        await member._retry_stuck_projects()
        await asyncio.gather(*list(member._inflight_retries))

        assert len(get_or_create_2pc_task_mock["calls"]) == 1
        pname, runner = get_or_create_2pc_task_mock["calls"][0]
        assert pname == "p1"
        assert isinstance(runner, functools.partial)
        assert runner.func == member._run_create_flow
        assert runner.args == (project, op_id)
        # The returned callable was actually scheduled and awaited.
        fake_callable.assert_awaited_once()
        # Update path was NOT taken.
        inline_runner_mock.assert_not_awaited()

    async def test_dispatches_deleting_via_background_task(
        self,
        crud_mock,
        session_helper_mock,
        get_or_create_2pc_task_mock,
        inline_runner_mock,
    ):
        # deleting → same shape as creating but with _run_delete_flow(name, op_id).
        op_id = uuid.uuid4()
        project = _make_stale_project(
            "p1", mlrun.common.schemas.ProjectState.deleting, op_id=op_id
        )
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[project])
        )
        fake_callable = unittest.mock.AsyncMock()
        get_or_create_2pc_task_mock["result"] = (fake_callable, "task-name")

        member = _make_member(leader_follower=crud_mock, followers={})
        await member._retry_stuck_projects()
        await asyncio.gather(*list(member._inflight_retries))

        assert len(get_or_create_2pc_task_mock["calls"]) == 1
        pname, runner = get_or_create_2pc_task_mock["calls"][0]
        assert pname == "p1"
        assert isinstance(runner, functools.partial)
        assert runner.func == member._run_delete_flow
        assert runner.args == ("p1", op_id)
        fake_callable.assert_awaited_once()
        inline_runner_mock.assert_not_awaited()

    async def test_dispatches_online_via_session_helper_and_inline_runner(
        self,
        crud_mock,
        session_helper_mock,
        get_or_create_2pc_task_mock,
        inline_runner_mock,
    ):
        # online → run_async_function_with_new_db_session(
        #     partial(run_project_2pc_runner_inline, runner, name)
        # ); the helper opens a session and the inline runner awaits the
        # update flow against it.
        op_id = uuid.uuid4()
        project = _make_stale_project(
            "p1", mlrun.common.schemas.ProjectState.online, op_id=op_id
        )
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[project])
        )

        member = _make_member(leader_follower=crud_mock, followers={})
        await member._retry_stuck_projects()
        await asyncio.gather(*list(member._inflight_retries))

        # session_helper_mock has the listing call (idx 0) and the update
        # dispatch (idx 1) in that order.
        assert len(session_helper_mock) == 2
        assert session_helper_mock[0][0] is crud_mock.list_stale_projects

        update_func = session_helper_mock[1][0]
        assert isinstance(update_func, functools.partial)
        # The outer partial is bound to the (now-mocked) inline runner with
        # (runner, project_name) — the helper appends db_session.
        assert update_func.func is framework.api.utils.run_project_2pc_runner_inline
        update_runner, update_name = update_func.args
        assert update_name == "p1"
        assert isinstance(update_runner, functools.partial)
        assert update_runner.func == member._run_update_flow
        assert update_runner.args == ("p1", project, op_id)

        # The inline runner was actually invoked (with the synthetic session
        # appended by session_helper_mock).
        inline_runner_mock.assert_awaited_once()
        # CREATE/DELETE path was NOT taken.
        assert get_or_create_2pc_task_mock["calls"] == []

    async def test_skip_when_active_task_exists(
        self,
        crud_mock,
        session_helper_mock,
        get_or_create_2pc_task_mock,
        inline_runner_mock,
    ):
        # task_callable=None means a sibling 2PC task is already running;
        # the retry must NOT spawn a duplicate.
        project = _make_stale_project("p1", mlrun.common.schemas.ProjectState.creating)
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[project])
        )
        get_or_create_2pc_task_mock["result"] = (None, "existing-task")

        member = _make_member(leader_follower=crud_mock, followers={})
        await member._retry_stuck_projects()

        assert member._inflight_retries == set()

    async def test_unknown_state_logs_and_does_nothing(
        self,
        crud_mock,
        session_helper_mock,
        get_or_create_2pc_task_mock,
        inline_runner_mock,
    ):
        # list_stale_projects shouldn't return rows in archived/offline, but
        # if it did (data drift, manual DB edit), we must not blindly fan
        # out — there's no flow for those states.
        project = _make_stale_project("p1", mlrun.common.schemas.ProjectState.archived)
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[project])
        )

        member = _make_member(leader_follower=crud_mock, followers={})
        await member._retry_stuck_projects()

        assert get_or_create_2pc_task_mock["calls"] == []
        inline_runner_mock.assert_not_awaited()
        assert member._inflight_retries == set()

    async def test_per_project_failure_does_not_block_others(
        self,
        crud_mock,
        session_helper_mock,
        inline_runner_mock,
        monkeypatch,
    ):
        # One project's dispatch raises synchronously — the loop must catch
        # it, log, and continue to the next project. The "good" one still
        # ends up scheduled.
        bad = _make_stale_project("bad", mlrun.common.schemas.ProjectState.creating)
        good = _make_stale_project("good", mlrun.common.schemas.ProjectState.creating)
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[bad, good])
        )

        good_callable = unittest.mock.AsyncMock()

        def factory(project_name, runner):
            if project_name == "bad":
                raise RuntimeError("dispatch failed")
            return good_callable, "task-name"

        monkeypatch.setattr(
            "framework.api.utils.get_or_create_project_2pc_background_task",
            factory,
        )

        member = _make_member(leader_follower=crud_mock, followers={})
        # MUST NOT raise — per-project failures are isolated.
        await member._retry_stuck_projects()
        await asyncio.gather(*list(member._inflight_retries))

        good_callable.assert_awaited_once()

    async def test_holds_task_reference_until_done(
        self,
        crud_mock,
        session_helper_mock,
        get_or_create_2pc_task_mock,
        inline_runner_mock,
    ):
        # Bare asyncio.create_task whose return is dropped can be GC'd
        # mid-run. Verify the inflight set holds the task while it's
        # pending and the done_callback discards it after completion.
        project = _make_stale_project("p1", mlrun.common.schemas.ProjectState.creating)
        crud_mock.list_stale_projects.return_value = (
            mlrun.common.schemas.ProjectsOutput(projects=[project])
        )

        # A callable that doesn't complete immediately — gives us a window
        # to observe the in-flight reference.
        gate = asyncio.Event()

        async def slow_callable():
            await gate.wait()

        get_or_create_2pc_task_mock["result"] = (slow_callable, "task-name")

        member = _make_member(leader_follower=crud_mock, followers={})
        await member._retry_stuck_projects()

        # While slow_callable is awaiting the gate, the task must be in
        # _inflight_retries.
        assert len(member._inflight_retries) == 1
        gate.set()
        await asyncio.gather(*list(member._inflight_retries))
        # done_callback runs after the awaitable completes, draining the set.
        assert member._inflight_retries == set()


class TestStartPeriodicSync:
    @pytest.fixture(autouse=True)
    def reset_2pc_gate(self):
        original = (
            mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc
        )
        try:
            yield
        finally:
            mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
                original
            )

    @pytest.fixture
    def captured_periodic(self, monkeypatch):
        captured: dict = {}

        def capture(interval, name, replace, function, *args, **kwargs):
            captured["interval"] = interval
            captured["name"] = name
            captured["replace"] = replace
            captured["function"] = function

        monkeypatch.setattr(
            "framework.utils.periodic.run_function_periodically",
            capture,
        )
        return captured

    def test_routes_to_retry_when_2pc_enabled(self, captured_periodic):
        mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
            "enabled"
        )
        member = _make_member(
            leader_follower=services.api.crud.Projects(),
            followers={"f": unittest.mock.Mock()},
        )
        member._periodic_sync_interval_seconds = 60

        member._start_periodic_sync()

        assert captured_periodic["function"] == member._retry_stuck_projects
        assert captured_periodic["name"] == "_retry_stuck_projects"
        assert captured_periodic["interval"] == 60

    def test_routes_to_legacy_when_2pc_disabled(self, captured_periodic):
        mlrun.mlconf.httpdb.clusterization.chief.feature_gates.project_sync_2pc = (
            "disabled"
        )
        member = _make_member(
            leader_follower=services.api.crud.Projects(),
            followers={"f": unittest.mock.Mock()},
        )
        member._periodic_sync_interval_seconds = 60

        member._start_periodic_sync()

        assert captured_periodic["function"] == member._sync_projects
        assert captured_periodic["name"] == "_sync_projects"

    def test_no_followers_skips_registration(self, captured_periodic):
        member = _make_member(
            leader_follower=services.api.crud.Projects(),
            followers={},
        )
        member._periodic_sync_interval_seconds = 60

        member._start_periodic_sync()

        assert captured_periodic == {}

    def test_zero_interval_skips_registration(self, captured_periodic):
        member = _make_member(
            leader_follower=services.api.crud.Projects(),
            followers={"f": unittest.mock.Mock()},
        )
        member._periodic_sync_interval_seconds = 0

        member._start_periodic_sync()

        assert captured_periodic == {}


class TestStopPeriodicSync:
    def test_cancels_both_possible_names(self, monkeypatch):
        # _start_periodic_sync registers exactly one of two names depending
        # on the 2PC gate. Shutdown can't know which (the gate may have
        # flipped), so it cancels both — extras are no-ops.
        cancelled: list[str] = []
        monkeypatch.setattr(
            "framework.utils.periodic.cancel_periodic_function",
            lambda name: cancelled.append(name),
        )

        member = _make_member(
            leader_follower=services.api.crud.Projects(), followers={}
        )
        member._stop_periodic_sync()

        assert "_sync_projects" in cancelled
        assert "_retry_stuck_projects" in cancelled
