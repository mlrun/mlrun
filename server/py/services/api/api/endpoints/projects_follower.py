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

import fastapi
import sqlalchemy.orm

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
from mlrun.utils import logger

import framework.api.deps
import framework.utils.clients.chief
import framework.utils.projects.follower_schemas as follower_schemas
import services.api.crud

router = fastapi.APIRouter()


def _to_follower_state(
    name: str, snapshot: mlrun.common.schemas.Project | None
) -> follower_schemas.FollowerProjectState:
    if snapshot is None:
        return follower_schemas.FollowerProjectState(name=name)
    return follower_schemas.FollowerProjectState(
        name=snapshot.metadata.name,
        op_id=snapshot.status.op_id,
        sync_status=snapshot.status.state,
    )


@router.post(
    "/follower/projects/{name}/prepare-create",
    response_model=follower_schemas.FollowerProjectState,
)
async def prepare_create_project(
    name: str,
    body: follower_schemas.FollowerPrepareCreateRequest,
) -> follower_schemas.FollowerProjectState:
    if body.metadata.name != name:
        raise mlrun.errors.MLRunBadRequestError(
            "Path project name does not match metadata.name"
        )
    project = mlrun.common.schemas.Project(
        metadata=body.metadata,
        spec=body.spec,
        status=mlrun.common.schemas.ProjectStatus(op_id=body.status.op_id),
    )
    result = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().prepare_create_project,
        project,
        body.status.op_id,
    )
    return _to_follower_state(name, result)


@router.post(
    "/follower/projects/{name}/commit-create",
    response_model=follower_schemas.FollowerProjectState,
)
async def commit_create_project(
    name: str,
    body: follower_schemas.FollowerCommitCreateRequest,
) -> follower_schemas.FollowerProjectState:
    result = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().commit_create_project, name, body.status.op_id
    )
    return _to_follower_state(name, result)


@router.put(
    "/follower/projects/{name}",
    response_model=follower_schemas.FollowerProjectState,
)
async def update_project(
    name: str,
    body: follower_schemas.FollowerUpdateRequest,
) -> follower_schemas.FollowerProjectState:
    if body.metadata.name != name:
        raise mlrun.errors.MLRunBadRequestError(
            "Path project name does not match metadata.name"
        )
    project = mlrun.common.schemas.Project(
        metadata=body.metadata,
        spec=body.spec,
        status=mlrun.common.schemas.ProjectStatus(op_id=body.status.op_id),
    )
    result = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().update_project_follower,
        name,
        project,
        body.status.op_id,
        body.prev_op_id,
    )
    return _to_follower_state(name, result)


@router.post(
    "/follower/projects/{name}/prepare-delete",
    response_model=follower_schemas.FollowerProjectState,
)
async def prepare_delete_project(
    name: str,
    body: follower_schemas.FollowerPrepareDeleteRequest,
) -> follower_schemas.FollowerProjectState:
    result = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().prepare_delete_project,
        name,
        body.status.op_id,
        body.prev_op_id,
    )
    return _to_follower_state(name, result)


@router.delete(
    "/follower/projects/{name}",
    response_model=follower_schemas.FollowerDeleteResult,
)
async def commit_delete_project(
    name: str,
    body: follower_schemas.FollowerCommitDeleteRequest,
    request: fastapi.Request,
) -> follower_schemas.FollowerDeleteResult:
    # delete_project_resources deletes schedules, which run only on chief, so we
    # re-route to chief — same reason the legacy (non-follower) delete endpoint does.
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to commit-delete follower project, re-routing to chief",
            project=name,
        )
        return (
            await framework.utils.clients.chief.Client().commit_delete_follower_project(
                name=name, request=request
            )
        )

    op_id = body.status.op_id
    # Validates internally (CAS/ordering/state) and, on success, purges the project's
    # resources and its row — a no-op if it's already gone (a previous call already
    # removed it) or a genuine retry with the same op_id (e.g. after a dropped
    # connection re-runs the purge rather than skipping it).
    await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().commit_delete_project, name, op_id
    )
    return follower_schemas.FollowerDeleteResult(
        name=name, op_id=op_id, result="removed"
    )


@router.get(
    "/follower/projects/states",
    response_model=follower_schemas.FollowerProjectStatesPage,
)
async def list_project_states(
    updated_after: datetime.datetime | None = None,
    cursor: str | None = None,
    page_size: int = fastapi.Query(200, alias="limit"),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> follower_schemas.FollowerProjectStatesPage:
    projects, next_cursor = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().list_project_states,
        db_session,
        updated_after,
        cursor,
        page_size,
    )
    return follower_schemas.FollowerProjectStatesPage(
        projects=[
            _to_follower_state(project.metadata.name, project) for project in projects
        ],
        next_cursor=next_cursor,
    )
