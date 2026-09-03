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
Wire request/response shapes for the leader -> follower 2PC project-sync contract.

Not in `mlrun.common.schemas`: that package is the public SDK<->server contract, and the
SDK never talks to this surface. Kept dependency-free like `follower_contract.py` (only
`pydantic` + `mlrun.common.schemas`, no FastAPI/SQLAlchemy) for the same reason — these
shapes are as portable as the validation rules and could move into a shared
cross-follower library alongside them.
"""

import uuid

import pydantic

import mlrun.common.schemas


class FollowerProjectState(pydantic.BaseModel):
    name: str
    op_id: uuid.UUID | None = None
    # Wire name is sync_status, not state, matching Orca's FollowerProjectState struct.
    sync_status: mlrun.common.schemas.ProjectState | None = None


class FollowerProjectStatesPage(pydantic.BaseModel):
    projects: list[FollowerProjectState]
    next_cursor: str | None = None


class FollowerOpIdStatus(pydantic.BaseModel):
    op_id: uuid.UUID


class FollowerPrepareCreateRequest(pydantic.BaseModel):
    metadata: mlrun.common.schemas.ProjectMetadata
    spec: mlrun.common.schemas.ProjectSpec = mlrun.common.schemas.ProjectSpec()
    status: FollowerOpIdStatus


class FollowerCommitCreateRequest(pydantic.BaseModel):
    status: FollowerOpIdStatus


class FollowerUpdateRequest(pydantic.BaseModel):
    metadata: mlrun.common.schemas.ProjectMetadata
    spec: mlrun.common.schemas.ProjectSpec = mlrun.common.schemas.ProjectSpec()
    status: FollowerOpIdStatus
    # None is a valid CAS witness for a project with no op_id yet (see
    # follower_contract.check_cas) — the leader omits this field entirely rather than
    # sending an explicit null when it has no witness to offer.
    prev_op_id: uuid.UUID | None = None


class FollowerPrepareDeleteRequest(pydantic.BaseModel):
    status: FollowerOpIdStatus
    prev_op_id: uuid.UUID | None = None


class FollowerCommitDeleteRequest(pydantic.BaseModel):
    status: FollowerOpIdStatus
