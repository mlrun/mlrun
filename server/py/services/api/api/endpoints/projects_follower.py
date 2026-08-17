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
The ML-12901 follower surface: leader (Orca) -> MLRun-as-follower calls.

Dedicated from the user-facing `/api/v1/projects` endpoints on purpose (see the Follower
Contract HLD) — this router takes no user token, only the configured leader's service
account (enforced by `deps.authenticate_leader_request`), so machine-origin auth never
mixes with user auth. Enterprise-only in practice: nothing calls this without an Orca
leader configured, but it stays always-registered rather than feature-gated.
"""

import datetime
import typing
import uuid

import fastapi
import pydantic
import sqlalchemy.orm
from fastapi import status

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils

import framework.api.deps
import framework.utils.background_tasks
import framework.utils.clients.chief
import framework.utils.projects.follower_contract as follower_contract
import services.api.crud

router = fastapi.APIRouter()


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
    result: typing.Literal["removed", "removal-scheduled"]


class FollowerPrepareCreateRequest(pydantic.BaseModel):
    project: mlrun.common.schemas.Project


class FollowerCommitCreateRequest(pydantic.BaseModel):
    op_id: uuid.UUID


class FollowerUpdateRequest(pydantic.BaseModel):
    project: mlrun.common.schemas.Project
    prev_op_id: uuid.UUID


class FollowerPrepareDeleteRequest(pydantic.BaseModel):
    op_id: uuid.UUID
    prev_op_id: uuid.UUID


class FollowerCommitDeleteRequest(pydantic.BaseModel):
    op_id: uuid.UUID


def _project_op_id(project: mlrun.common.schemas.Project) -> uuid.UUID:
    if project.status.op_id is None:
        raise mlrun.errors.MLRunBadRequestError("project.status.op_id is required")
    return project.status.op_id


def _to_follower_state(
    name: str, snapshot: mlrun.common.schemas.Project | None
) -> FollowerProjectState:
    if snapshot is None:
        return FollowerProjectState(name=name)
    return FollowerProjectState(
        name=snapshot.metadata.name,
        op_id=snapshot.status.op_id,
        state=snapshot.status.state,
    )


async def _reroute_to_chief_if_worker(
    request: fastapi.Request, method: str, path: str
) -> fastapi.Response | None:
    """
    Used only by `commit_delete_project` — the one route in this router that touches
    `InternalBackgroundTasksHandler`, which is chief-only. A worker replica re-routes
    that call to chief rather than handling it itself.
    """
    if (
        mlrun.mlconf.httpdb.clusterization.role
        == mlrun.common.schemas.ClusterizationRole.chief
    ):
        return None
    return await framework.utils.clients.chief.Client().proxy_follower_project_request(
        method, path, request
    )


@router.post(
    "/follower/projects/{name}/prepare-create",
    response_model=FollowerProjectState,
)
async def prepare_create_project(
    name: str,
    body: FollowerPrepareCreateRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> FollowerProjectState:
    project = body.project
    if project.metadata.name != name:
        raise mlrun.errors.MLRunBadRequestError(
            "Path project name does not match project.metadata.name"
        )
    op_id = _project_op_id(project)
    await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().prepare_create_project, project, op_id
    )
    snapshot = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().get_follower_project_snapshot, db_session, name
    )
    return _to_follower_state(name, snapshot)


@router.post(
    "/follower/projects/{name}/commit-create",
    response_model=FollowerProjectState,
)
async def commit_create_project(
    name: str,
    body: FollowerCommitCreateRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> FollowerProjectState:
    await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().commit_create_project, name, body.op_id
    )
    snapshot = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().get_follower_project_snapshot, db_session, name
    )
    return _to_follower_state(name, snapshot)


@router.put(
    "/follower/projects/{name}",
    response_model=FollowerProjectState,
)
async def update_project(
    name: str,
    body: FollowerUpdateRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> FollowerProjectState:
    project = body.project
    if project.metadata.name != name:
        raise mlrun.errors.MLRunBadRequestError(
            "Path project name does not match project.metadata.name"
        )
    op_id = _project_op_id(project)
    await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().update_project_follower,
        name,
        project,
        op_id,
        body.prev_op_id,
    )
    snapshot = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().get_follower_project_snapshot, db_session, name
    )
    return _to_follower_state(name, snapshot)


@router.post(
    "/follower/projects/{name}/prepare-delete",
    response_model=FollowerProjectState,
)
async def prepare_delete_project(
    name: str,
    body: FollowerPrepareDeleteRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> FollowerProjectState:
    await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().prepare_delete_project,
        name,
        body.op_id,
        body.prev_op_id,
    )
    snapshot = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().get_follower_project_snapshot, db_session, name
    )
    return _to_follower_state(name, snapshot)


@router.delete(
    "/follower/projects/{name}",
    response_model=FollowerDeleteResult,
)
async def commit_delete_project(
    name: str,
    body: FollowerCommitDeleteRequest,
    response: fastapi.Response,
    request: fastapi.Request,
    background_tasks: fastapi.BackgroundTasks,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> FollowerDeleteResult:
    # Only this endpoint reroutes to chief: it's the one call in this router that
    # touches InternalBackgroundTasksHandler, which is chief-only — the other 4 do
    # plain project DB writes, same as the legacy (non-rerouted) project CUD endpoints.
    if proxied := await _reroute_to_chief_if_worker(
        request, "DELETE", f"follower/projects/{name}"
    ):
        return proxied

    op_id = body.op_id
    snapshot = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().get_follower_project_snapshot, db_session, name
    )
    if snapshot is None:
        # Already fully removed by a previous call — idempotent no-op per the contract.
        return FollowerDeleteResult(name=name, op_id=op_id, result="removed")

    # Fail fast on an invalid call (stale/mismatched op_id, wrong state) before
    # scheduling anything — commit_delete_project() re-runs this same check when the
    # background task actually executes, which is then a safe, cheap no-op.
    follower_contract.validate_call(
        follower_contract.FollowerOp.commit_delete,
        current_state=snapshot.status.state,
        stored_op_id=snapshot.status.op_id,
        incoming_op_id=op_id,
    )

    kind = framework.utils.background_tasks.BackgroundTaskKinds.project_deletion.format(
        name
    )
    handler = framework.utils.background_tasks.InternalBackgroundTasksHandler()
    existing_task = handler.get_active_background_task_by_kind(kind)
    response.status_code = status.HTTP_202_ACCEPTED
    if existing_task is None:
        # No active task for this project: either the first call, or the previous
        # attempt already finished (`InternalBackgroundTasksHandler` clears the active
        # slot on both success and failure) and `snapshot` above being non-None means
        # that attempt failed — either way, kicking off a fresh one is the right retry.
        task, _ = handler.create_background_task(
            kind,
            mlrun.mlconf.background_tasks.default_timeouts.operations.delete_project,
            services.api.crud.Projects().commit_delete_project,
            None,  # background task's own id — let it generate one
            name,
            op_id,
        )
        background_tasks.add_task(task)
    # else: an active task exists and is still running (a just-succeeded/failed task
    # would already have been cleared from the active slot, per the note above).
    return FollowerDeleteResult(name=name, op_id=op_id, result="removal-scheduled")


@router.get(
    "/follower/projects/states",
    response_model=FollowerProjectStatesPage,
)
async def list_project_states(
    updated_after: datetime.datetime | None = None,
    cursor: str | None = None,
    page_size: int = 200,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> FollowerProjectStatesPage:
    projects, next_cursor = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().list_project_states,
        db_session,
        updated_after,
        cursor,
        page_size,
    )
    return FollowerProjectStatesPage(
        projects=[
            _to_follower_state(project.metadata.name, project) for project in projects
        ],
        next_cursor=next_cursor,
    )
