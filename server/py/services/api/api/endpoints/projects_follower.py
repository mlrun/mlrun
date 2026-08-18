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
import uuid

import fastapi
import sqlalchemy.orm

import mlrun
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils

import framework.api.deps
import framework.utils.clients.chief
import framework.utils.projects.follower_schemas as follower_schemas
import services.api.crud

router = fastapi.APIRouter()


def _project_op_id(project: mlrun.common.schemas.Project) -> uuid.UUID:
    if project.status.op_id is None:
        raise mlrun.errors.MLRunBadRequestError("project.status.op_id is required")
    return project.status.op_id


def _to_follower_state(
    name: str, snapshot: mlrun.common.schemas.Project | None
) -> follower_schemas.FollowerProjectState:
    if snapshot is None:
        return follower_schemas.FollowerProjectState(name=name)
    return follower_schemas.FollowerProjectState(
        name=snapshot.metadata.name,
        op_id=snapshot.status.op_id,
        state=snapshot.status.state,
    )


async def _reroute_to_chief_if_worker(
    request: fastapi.Request, method: str, path: str
) -> fastapi.Response | None:
    """
    Used only by `commit_delete_project` — the one route in this router whose work
    (`delete_project_resources`, via `delete_schedules`) is chief-only. A worker
    replica re-routes that call to chief rather than handling it itself.
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
    response_model=follower_schemas.FollowerProjectState,
)
async def prepare_create_project(
    name: str,
    body: follower_schemas.FollowerPrepareCreateRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> follower_schemas.FollowerProjectState:
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
    response_model=follower_schemas.FollowerProjectState,
)
async def commit_create_project(
    name: str,
    body: follower_schemas.FollowerCommitCreateRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> follower_schemas.FollowerProjectState:
    await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().commit_create_project, name, body.op_id
    )
    snapshot = await mlrun.utils.run_in_threadpool(
        services.api.crud.Projects().get_follower_project_snapshot, db_session, name
    )
    return _to_follower_state(name, snapshot)


@router.put(
    "/follower/projects/{name}",
    response_model=follower_schemas.FollowerProjectState,
)
async def update_project(
    name: str,
    body: follower_schemas.FollowerUpdateRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> follower_schemas.FollowerProjectState:
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
    response_model=follower_schemas.FollowerProjectState,
)
async def prepare_delete_project(
    name: str,
    body: follower_schemas.FollowerPrepareDeleteRequest,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> follower_schemas.FollowerProjectState:
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
    response_model=follower_schemas.FollowerDeleteResult,
)
async def commit_delete_project(
    name: str,
    body: follower_schemas.FollowerCommitDeleteRequest,
    request: fastapi.Request,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
) -> follower_schemas.FollowerDeleteResult:
    # Deliberately blocking, per Orca's requirement: this call does not return until
    # the purge has actually finished — no background task, no early 202. Still
    # reroutes to chief: delete_project_resources() deletes schedules, which are
    # chief-only, same reason the legacy (non-follower) delete endpoint reroutes.
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
        return follower_schemas.FollowerDeleteResult(
            name=name, op_id=op_id, result="removed"
        )

    # Validates internally (CAS/ordering/state) and, on success, purges the project's
    # resources and its row before returning — a genuine retry with the same op_id
    # (e.g. after a dropped connection) re-runs the purge rather than skipping it.
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
    page_size: int = 200,
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
