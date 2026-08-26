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

import datetime
import uuid

import pytest
from uuid_utils.compat import uuid7

import mlrun.common.schemas
import mlrun.errors

import framework.utils.projects.follower_contract as follower_contract

_OLD = uuid.UUID(int=50)
_STORED = uuid.UUID(int=100)
_NEW = uuid.UUID(int=200)

_STATE = mlrun.common.schemas.ProjectState


# ----- op_id_timestamp -------------------------------------------------------


def test_op_id_timestamp_extracts_the_embedded_mint_time():
    before = datetime.datetime.now(tz=datetime.UTC)
    op_id = uuid7()
    after = datetime.datetime.now(tz=datetime.UTC)

    timestamp = follower_contract.op_id_timestamp(op_id)

    assert before - datetime.timedelta(milliseconds=1) <= timestamp <= after


def test_op_id_timestamp_rejects_a_non_v7_uuid():
    with pytest.raises(ValueError, match="UUIDv7"):
        follower_contract.op_id_timestamp(_STORED)


# ----- check_ordering -------------------------------------------------------


def test_check_ordering_applies_on_first_call_with_no_stored_op():
    assert (
        follower_contract.check_ordering(None, _STORED)
        == follower_contract.ReplayOutcome.apply
    )


def test_check_ordering_applies_on_newer_incoming_op():
    assert (
        follower_contract.check_ordering(_STORED, _NEW)
        == follower_contract.ReplayOutcome.apply
    )


def test_check_ordering_replays_on_equal_op():
    assert (
        follower_contract.check_ordering(_STORED, _STORED)
        == follower_contract.ReplayOutcome.replay
    )


def test_check_ordering_rejects_stale_incoming_op():
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.check_ordering(_STORED, _OLD)


# ----- check_cas -------------------------------------------------------------


def test_check_cas_rejects_missing_witness_against_a_known_op():
    """A project MLRun already has an op_id for, but the caller sends no witness at
    all, is a real CAS mismatch (409) — not a special "missing witness" 400. Only
    `stored == prev_op_id` matters, and `None != _STORED`."""
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.check_cas(_STORED, None)


def test_check_cas_rejects_mismatched_witness():
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.check_cas(_STORED, _OLD)


def test_check_cas_accepts_matching_witness():
    follower_contract.check_cas(_STORED, _STORED)  # does not raise


def test_check_cas_accepts_no_witness_against_a_project_with_no_prior_op():
    """A project that existed before this follower interface has op_id=NULL — the
    leader's first touch of it legitimately has no witness to offer either, and that
    must be accepted, not rejected as a missing-witness error."""
    follower_contract.check_cas(None, None)  # does not raise


# ----- check_same_op (commit_create / commit_delete) ------------------------


def test_check_same_op_rejects_when_nothing_in_flight():
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.check_same_op(None, _STORED)


def test_check_same_op_rejects_mismatched_op():
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.check_same_op(_STORED, _NEW)


def test_check_same_op_replays_on_matching_op():
    assert (
        follower_contract.check_same_op(_STORED, _STORED)
        == follower_contract.ReplayOutcome.apply
    )


# ----- check_transition: per-op valid/invalid starting states ---------------


@pytest.mark.parametrize(
    "current_state",
    [None, _STATE.creating],
)
def test_check_transition_prepare_create_accepts_valid_states(current_state):
    follower_contract.check_transition(
        current_state, follower_contract.FollowerOp.prepare_create
    )


@pytest.mark.parametrize(
    "current_state",
    [_STATE.online, _STATE.deleting, _STATE.archived],
)
def test_check_transition_prepare_create_rejects_invalid_states(current_state):
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.check_transition(
            current_state, follower_contract.FollowerOp.prepare_create
        )


@pytest.mark.parametrize("current_state", [_STATE.creating, _STATE.online])
def test_check_transition_commit_create_accepts_valid_states(current_state):
    follower_contract.check_transition(
        current_state, follower_contract.FollowerOp.commit_create
    )


@pytest.mark.parametrize("current_state", [None, _STATE.deleting, _STATE.archived])
def test_check_transition_commit_create_rejects_invalid_states(current_state):
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.check_transition(
            current_state, follower_contract.FollowerOp.commit_create
        )


@pytest.mark.parametrize("current_state", [_STATE.online, _STATE.archived])
def test_check_transition_update_accepts_valid_states(current_state):
    follower_contract.check_transition(
        current_state, follower_contract.FollowerOp.update
    )


def test_check_transition_update_rejects_absent_project_as_not_found():
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        follower_contract.check_transition(None, follower_contract.FollowerOp.update)


@pytest.mark.parametrize("current_state", [_STATE.creating, _STATE.deleting])
def test_check_transition_update_rejects_other_invalid_states(current_state):
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.check_transition(
            current_state, follower_contract.FollowerOp.update
        )


@pytest.mark.parametrize(
    "current_state", [None, _STATE.online, _STATE.archived, _STATE.deleting]
)
def test_check_transition_prepare_delete_accepts_valid_states(current_state):
    follower_contract.check_transition(
        current_state, follower_contract.FollowerOp.prepare_delete
    )


def test_check_transition_prepare_delete_rejects_creating():
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.check_transition(
            _STATE.creating, follower_contract.FollowerOp.prepare_delete
        )


@pytest.mark.parametrize("current_state", [None, _STATE.deleting])
def test_check_transition_commit_delete_accepts_valid_states(current_state):
    follower_contract.check_transition(
        current_state, follower_contract.FollowerOp.commit_delete
    )


@pytest.mark.parametrize(
    "current_state", [_STATE.creating, _STATE.online, _STATE.archived]
)
def test_check_transition_commit_delete_rejects_invalid_states(current_state):
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.check_transition(
            current_state, follower_contract.FollowerOp.commit_delete
        )


# ----- validate_call: integrated per-op sequences ----------------------------


def test_validate_call_prepare_create_first_call_applies():
    outcome = follower_contract.validate_call(
        follower_contract.FollowerOp.prepare_create,
        current_state=None,
        stored_op_id=None,
        incoming_op_id=_STORED,
    )
    assert outcome == follower_contract.ReplayOutcome.apply


def test_validate_call_prepare_create_replay_after_commit_create_does_not_reject():
    """
    Regression: a prepare-create retry (lost response, leader re-sends the exact same
    op_id) arriving after commit-create has already flipped the project to `online`
    must be treated as a safe replay, not rejected for being in the "wrong" state.
    """
    outcome = follower_contract.validate_call(
        follower_contract.FollowerOp.prepare_create,
        current_state=_STATE.online,
        stored_op_id=_STORED,
        incoming_op_id=_STORED,
    )
    assert outcome == follower_contract.ReplayOutcome.replay


def test_validate_call_prepare_create_new_op_on_online_project_still_rejected():
    """
    The replay-skips-transition-check fix must not open a hole: a genuinely new
    (newer, non-matching) create op_id arriving on an already-online project is a
    real "already exists" conflict, not a replay, and must still be rejected.
    """
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.validate_call(
            follower_contract.FollowerOp.prepare_create,
            current_state=_STATE.online,
            stored_op_id=_STORED,
            incoming_op_id=_NEW,
        )


def test_validate_call_commit_create_requires_provisioned():
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        follower_contract.validate_call(
            follower_contract.FollowerOp.commit_create,
            current_state=None,
            stored_op_id=None,
            incoming_op_id=_STORED,
        )


def test_validate_call_update_first_touch_with_no_prior_op_id_applies():
    """Migration scenario: a project that predates this follower interface has
    op_id=None; the leader observes that and sends prev_op_id=None to match — this
    must apply, not be rejected as a missing witness."""
    outcome = follower_contract.validate_call(
        follower_contract.FollowerOp.update,
        current_state=_STATE.online,
        stored_op_id=None,
        incoming_op_id=_STORED,
        prev_op_id=None,
    )
    assert outcome == follower_contract.ReplayOutcome.apply


def test_validate_call_update_cas_mismatch_rejected():
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.validate_call(
            follower_contract.FollowerOp.update,
            current_state=_STATE.online,
            stored_op_id=_STORED,
            incoming_op_id=_NEW,
            prev_op_id=_OLD,
        )


def test_validate_call_update_stale_ordering_rejected():
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.validate_call(
            follower_contract.FollowerOp.update,
            current_state=_STATE.online,
            stored_op_id=_STORED,
            incoming_op_id=_OLD,
            prev_op_id=_STORED,
        )


def test_validate_call_update_valid_cas_and_ordering_applies():
    outcome = follower_contract.validate_call(
        follower_contract.FollowerOp.update,
        current_state=_STATE.online,
        stored_op_id=_STORED,
        incoming_op_id=_NEW,
        prev_op_id=_STORED,
    )
    assert outcome == follower_contract.ReplayOutcome.apply


def test_validate_call_prepare_delete_replay_of_in_flight_delete():
    """A repeated mark-delete call (leader resends the same op) while already
    `deleting` is an idempotent no-op, not an error."""
    outcome = follower_contract.validate_call(
        follower_contract.FollowerOp.prepare_delete,
        current_state=_STATE.deleting,
        stored_op_id=_STORED,
        incoming_op_id=_STORED,
        prev_op_id=_STORED,
    )
    assert outcome == follower_contract.ReplayOutcome.replay


def test_validate_call_commit_delete_matching_op_applies():
    outcome = follower_contract.validate_call(
        follower_contract.FollowerOp.commit_delete,
        current_state=_STATE.deleting,
        stored_op_id=_STORED,
        incoming_op_id=_STORED,
    )
    assert outcome == follower_contract.ReplayOutcome.apply


def test_validate_call_commit_delete_mismatched_op_rejected():
    with pytest.raises(mlrun.errors.MLRunConflictError):
        follower_contract.validate_call(
            follower_contract.FollowerOp.commit_delete,
            current_state=_STATE.deleting,
            stored_op_id=_STORED,
            incoming_op_id=_NEW,
        )
