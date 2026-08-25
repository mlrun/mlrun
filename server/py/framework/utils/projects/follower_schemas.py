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
Wire request/response shapes for the leader -> follower 2PC project-sync contract (see
the "Follower Contract HLD" Confluence page, ML-12901).

Not in `mlrun.common.schemas`: that package is the public SDK<->server contract, and the
SDK never talks to this surface. Kept dependency-free like `follower_contract.py` (only
`pydantic` + `mlrun.common.schemas`, no FastAPI/SQLAlchemy) for the same reason — these
shapes are as portable as the validation rules and could move into a shared
cross-follower library alongside them.
"""

import datetime
import typing
import uuid

import pydantic

import mlrun.common.schemas


class FollowerProjectState(pydantic.BaseModel):
    name: str
    op_id: uuid.UUID | None = None
    state: mlrun.common.schemas.ProjectState | None = None


class FollowerProjectStatesPage(pydantic.BaseModel):
    projects: list[FollowerProjectState]
    next_cursor: str | None = None


class FollowerDeleteResult(pydantic.BaseModel):
    name: str
    op_id: uuid.UUID
    # Always "removed": commit-delete blocks until the purge actually finishes (Orca
    # requirement) — there is no async/scheduled outcome to represent. A call that
    # can't complete raises an HTTP error instead of returning a soft in-progress state.
    result: typing.Literal["removed"]


class FollowerPrepareCreateRequest(pydantic.BaseModel):
    project: mlrun.common.schemas.Project


class FollowerCommitCreateRequest(pydantic.BaseModel):
    op_id: uuid.UUID
    updated_at: datetime.datetime


class FollowerUpdateRequest(pydantic.BaseModel):
    project: mlrun.common.schemas.Project
    # Optional: None is a valid CAS witness for a project with no op_id yet (see
    # follower_contract.check_cas). Required-but-nullable, not omittable — the leader
    # always sends this field, its value is what can legitimately be None.
    prev_op_id: uuid.UUID | None = None


class FollowerPrepareDeleteRequest(pydantic.BaseModel):
    op_id: uuid.UUID
    updated_at: datetime.datetime
    prev_op_id: uuid.UUID | None = None


class FollowerCommitDeleteRequest(pydantic.BaseModel):
    op_id: uuid.UUID
