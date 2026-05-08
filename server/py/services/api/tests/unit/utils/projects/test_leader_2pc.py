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

import functools
import unittest.mock
import uuid

import pytest

import mlrun.common.schemas
import mlrun.config
import mlrun.errors

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
    return member


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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

        await member._run_create_flow(project, op_id, db_session="sess")

        # Full ordered pipeline: prepare → advance → commit → complete.
        follower.prepare_create_project.assert_called_once_with(project, op_id)
        crud_mock.advance_create_project_to_commit.assert_called_once_with(
            "sess", "p1", op_id
        )
        follower.commit_create_project.assert_called_once_with("p1", op_id)
        crud_mock.complete_create_project.assert_called_once_with(
            "sess", "p1", op_id
        )

    async def test_phase_one_resumes_only_commit(self, crud_mock):
        # Resume-from-crash path: row is at phase=1 because a previous run
        # finished `advance` but never reached `complete`.
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 1
        follower = unittest.mock.Mock()

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

        with pytest.raises(ExceptionGroup):
            await member._run_create_flow(project, op_id, db_session="sess")

        crud_mock.complete_create_project.assert_not_called()


class TestRunUpdateFlow:
    async def test_phase_zero_fans_out_then_completes(self, crud_mock):
        op_id = uuid.uuid4()
        project = _make_project()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

        await member._run_update_flow("p1", project, op_id, db_session="sess")

        follower.update_project_follower.assert_called_once_with(
            "p1", project, op_id
        )
        crud_mock.complete_update_project.assert_called_once_with(
            "sess", "p1", op_id
        )

    async def test_phase_none_skips_everything(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = None
        follower = unittest.mock.Mock()

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

        await member._run_update_flow(
            "p1", _make_project(), op_id, db_session="sess"
        )

        follower.update_project_follower.assert_not_called()
        crud_mock.complete_update_project.assert_not_called()

    async def test_follower_failure_blocks_complete(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = 0
        follower = unittest.mock.Mock()
        follower.update_project_follower.side_effect = RuntimeError("boom")

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )

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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )
        # post_delete_project is an async hook on the base member; stub it so
        # we can assert it ran exactly once after the row was removed.
        member.post_delete_project = unittest.mock.AsyncMock()

        await member._run_delete_flow("p1", op_id, db_session="sess")

        follower.prepare_delete_project.assert_called_once_with("p1", op_id)
        crud_mock.advance_delete_project_to_commit.assert_called_once_with(
            "sess", "p1", op_id
        )
        follower.commit_delete_project.assert_called_once_with("p1", op_id)
        crud_mock.complete_delete_project.assert_called_once_with(
            "sess", "p1", op_id
        )
        # post_delete_project must run only after the row is gone — i.e. only
        # if complete_delete_project was reached.
        member.post_delete_project.assert_awaited_once_with("p1")

    async def test_phase_one_skips_prepare_and_advance(self, crud_mock):
        op_id = uuid.uuid4()
        crud_mock.get_project_sync_phase.return_value = 1
        follower = unittest.mock.Mock()

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )
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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )
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

        member = _make_member(
            leader_follower=crud_mock, followers={"nuc": follower}
        )
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
