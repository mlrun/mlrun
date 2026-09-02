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

"""
CAS + UUIDv7-ordering + state-machine checks for the leader -> follower 2PC project-sync
contract.

Deliberately dependency-free (no DB session, no FastAPI, no SQLAlchemy types) so the
contract logic stays decoupled from MLRun's own storage/transport and can be lifted into a
shared cross-follower library later without rework. Callers translate their own storage
(DB row, CRD labels, ...) into plain `op_id`/`ProjectState` values before calling in, and
back again afterward.
"""

import enum
import uuid

import mlrun.common.schemas
import mlrun.errors


class FollowerOp(enum.Enum):
    """The six leader -> follower operations in the project-sync 2PC contract."""

    prepare_create = "prepare_create"
    commit_create = "commit_create"
    update = "update"
    prepare_delete = "prepare_delete"
    commit_delete = "commit_delete"


class ReplayOutcome(enum.Enum):
    apply = "apply"
    replay = "replay"


# Ops that carry a `prev_op_id` CAS witness distinct from their own new `op_id`.
_CAS_OPS = frozenset({FollowerOp.update, FollowerOp.prepare_delete})
# Ops that reuse the same op_id minted by their matching prepare step (equality, not
# ordering) rather than a freshly minted one.
_SAME_OP_OPS = frozenset({FollowerOp.commit_create, FollowerOp.commit_delete})

# Valid current states per op, per the Follower Contract's request grid. `None` means "no
# project row exists yet" and is listed explicitly where that's a valid starting point.
_VALID_STATES: dict[FollowerOp, frozenset[mlrun.common.schemas.ProjectState | None]] = {
    FollowerOp.prepare_create: frozenset(
        {None, mlrun.common.schemas.ProjectState.creating}
    ),
    FollowerOp.commit_create: frozenset(
        {
            mlrun.common.schemas.ProjectState.creating,
            mlrun.common.schemas.ProjectState.online,
        }
    ),
    FollowerOp.update: frozenset(
        {
            mlrun.common.schemas.ProjectState.online,
            mlrun.common.schemas.ProjectState.archived,
        }
    ),
    FollowerOp.prepare_delete: frozenset(
        {
            None,
            mlrun.common.schemas.ProjectState.online,
            mlrun.common.schemas.ProjectState.archived,
            mlrun.common.schemas.ProjectState.deleting,
        }
    ),
    FollowerOp.commit_delete: frozenset(
        {None, mlrun.common.schemas.ProjectState.deleting}
    ),
}


def check_cas(stored_op_id: uuid.UUID | None, prev_op_id: uuid.UUID | None) -> None:
    """
    Enforce the CAS witness for `update` and `prepare_delete`: the caller's `prev_op_id`
    must match what's currently stored. `prepare_create`/`commit_create`/`commit_delete`
    don't call this — they have no separate witness (see `check_ordering`/`check_same_op`).

    `None == None` is a valid match, not a missing-witness error: a project that existed
    before this follower interface has `op_id = NULL` (nothing auto-fills it at migration
    time), so the leader's *first* touch of such a project legitimately has no witness to
    offer either — it observes `None` via `list_projects` and sends that back as
    `prev_op_id`. (Rejecting a missing `prev_op_id` with 400 is a client<->leader boundary
    rule; there is no such rule for leader<->follower.)
    """
    if stored_op_id != prev_op_id:
        raise mlrun.errors.MLRunConflictError(
            f"CAS mismatch: stored op_id {stored_op_id} does not match "
            f"prev_op_id {prev_op_id}"
        )


def check_ordering(
    stored_op_id: uuid.UUID | None, incoming_op_id: uuid.UUID
) -> ReplayOutcome:
    """
    Enforce UUIDv7 time-ordering for `prepare_create` and `update`/`prepare_delete` (after
    CAS passes): the incoming op_id must be newer than, or equal to, what's stored.

    Equal op_id is an idempotent replay (safe no-op). An incoming op_id older than stored
    is a stale/out-of-order call and is rejected.
    """
    if stored_op_id is None or incoming_op_id > stored_op_id:
        return ReplayOutcome.apply
    if incoming_op_id == stored_op_id:
        return ReplayOutcome.replay
    raise mlrun.errors.MLRunConflictError(
        f"Stale operation: incoming op_id {incoming_op_id} is older than "
        f"stored op_id {stored_op_id}"
    )


def check_same_op(
    stored_op_id: uuid.UUID | None, incoming_op_id: uuid.UUID
) -> ReplayOutcome:
    """
    For `commit_create`/`commit_delete`: the incoming op_id must equal the in-flight op
    exactly. Unlike `update`/`prepare_delete`, these steps reuse the op_id minted by their
    matching prepare step rather than a freshly minted one, so this is an equality check,
    not an ordering one.
    """
    if stored_op_id is None:
        raise mlrun.errors.MLRunPreconditionFailedError(
            "No in-flight operation to commit against"
        )
    if incoming_op_id == stored_op_id:
        return ReplayOutcome.apply
    raise mlrun.errors.MLRunConflictError(
        f"op_id mismatch: incoming {incoming_op_id} does not match "
        f"the in-flight op {stored_op_id}"
    )


def check_transition(
    current_state: mlrun.common.schemas.ProjectState | None, op: FollowerOp
) -> None:
    """
    Validate that `current_state` allows `op`, per the Follower Contract's request grid.
    Raises the error class the contract's status-code table calls for on an invalid
    transition. Does not distinguish "apply" from "no-op replay" — that's `check_ordering`/
    `check_same_op`'s job; this only rejects transitions that are never valid.
    """
    if current_state in _VALID_STATES[op]:
        # Covers, among others, the absent-project (current_state=None) case for
        # prepare_delete/commit_delete: that's a valid no-op starting point for both,
        # not an error, per the contract's absent-project semantics.
        return
    # The contract specifies a different status per op for "doesn't exist": update -> 404
    # (explicit in its status table); everything else that reaches here (e.g. commit_create
    # on an absent project) falls through to the generic 412 below, per its own status
    # table (403/409/412, no 404). Don't generalize this to "state is None -> 404" — that
    # would wrongly turn commit_create's 412 into a 404.
    #
    # Unreachable from MLRun's own call path today: crud.Projects.update_project_follower
    # already raises 404 before calling validate_call when the project is absent, so
    # current_state can't be None here for update in practice. Kept anyway as the
    # contract's own source of truth for a caller that doesn't pre-guard absence the
    # way MLRun's crud layer happens to (e.g. a future follower implementation).
    if op == FollowerOp.update and current_state is None:
        raise mlrun.errors.MLRunNotFoundError("Project not found")
    raise mlrun.errors.MLRunPreconditionFailedError(
        f"Project is not in a valid state for {op.value} (state={current_state})"
    )


def validate_call(
    op: FollowerOp,
    current_state: mlrun.common.schemas.ProjectState | None,
    stored_op_id: uuid.UUID | None,
    incoming_op_id: uuid.UUID,
    prev_op_id: uuid.UUID | None = None,
) -> ReplayOutcome:
    """
    Run the full per-call validation order the contract specifies: CAS -> ordering ->
    valid transition. Returns whether the caller should apply the mutation or treat this
    as an idempotent no-op replay.

    `check_transition` only runs when the outcome is `apply`. A `replay` means the
    incoming op_id exactly matches what's stored, i.e. nothing has changed since that
    exact op was already validated and applied — the current state may have since moved
    on via a later, unrelated op (e.g. a `prepare_create` retry arriving after
    `commit_create` already flipped the project to `online`), so re-running a transition
    check meant for a *new* mutation against today's state would reject a legitimate
    replay of yesterday's already-valid one.
    """
    if op in _CAS_OPS:
        check_cas(stored_op_id, prev_op_id)
        outcome = check_ordering(stored_op_id, incoming_op_id)
    elif op in _SAME_OP_OPS:
        outcome = check_same_op(stored_op_id, incoming_op_id)
    else:
        outcome = check_ordering(stored_op_id, incoming_op_id)
    if outcome == ReplayOutcome.apply:
        check_transition(current_state, op)
    return outcome
