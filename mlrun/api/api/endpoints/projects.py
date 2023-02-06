# Copyright 2018 Iguazio
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
#
import http
import typing

import fastapi
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.chief
from mlrun.api.utils.singletons.project_member import get_project_member
from mlrun.utils import logger

router = fastapi.APIRouter()


@router.post(
    "/projects",
    responses={
        http.HTTPStatus.CREATED.value: {"model": mlrun.api.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def create_project(
    project: mlrun.api.schemas.Project,
    response: fastapi.Response,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    project, is_running_in_background = get_project_member().create_project(
        db_session,
        project,
        auth_info.projects_role,
        auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    response.status_code = http.HTTPStatus.CREATED.value
    return project


@router.put(
    "/projects/{name}",
    responses={
        http.HTTPStatus.OK.value: {"model": mlrun.api.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def store_project(
    project: mlrun.api.schemas.Project,
    name: str,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    project, is_running_in_background = get_project_member().store_project(
        db_session,
        name,
        project,
        auth_info.projects_role,
        auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return project


@router.patch(
    "/projects/{name}",
    responses={
        http.HTTPStatus.OK.value: {"model": mlrun.api.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def patch_project(
    project: dict,
    name: str,
    patch_mode: mlrun.api.schemas.PatchMode = fastapi.Header(
        mlrun.api.schemas.PatchMode.replace,
        alias=mlrun.api.schemas.HeaderNames.patch_mode,
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    project, is_running_in_background = get_project_member().patch_project(
        db_session,
        name,
        project,
        patch_mode,
        auth_info.projects_role,
        auth_info.session,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return project


@router.get("/projects/{name}", response_model=mlrun.api.schemas.Project)
async def get_project(
    name: str,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    project = await run_in_threadpool(
        get_project_member().get_project, db_session, name, auth_info.session
    )
    # skip permission check if it's the leader
    if not _is_request_from_leader(auth_info.projects_role):
        await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
    return project


@router.delete(
    "/projects/{name}",
    responses={
        http.HTTPStatus.NO_CONTENT.value: {},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
async def delete_project(
    name: str,
    request: fastapi.Request,
    deletion_strategy: mlrun.api.schemas.DeletionStrategy = fastapi.Header(
        mlrun.api.schemas.DeletionStrategy.default(),
        alias=mlrun.api.schemas.HeaderNames.deletion_strategy,
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    # delete project can be responsible for deleting schedules. Schedules are running only on chief,
    # that is why we re-route requests to chief
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to delete project, re-routing to chief",
            project=name,
            deletion_strategy=deletion_strategy,
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return await chief_client.delete_project(name=name, request=request)

    is_running_in_background = await run_in_threadpool(
        get_project_member().delete_project,
        db_session,
        name,
        deletion_strategy,
        auth_info.projects_role,
        auth_info,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


@router.get("/projects", response_model=mlrun.api.schemas.ProjectsOutput)
async def list_projects(
    format_: mlrun.api.schemas.ProjectsFormat = fastapi.Query(
        mlrun.api.schemas.ProjectsFormat.full, alias="format"
    ),
    owner: str = None,
    labels: typing.List[str] = fastapi.Query(None, alias="label"),
    state: mlrun.api.schemas.ProjectState = None,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    allowed_project_names = None
    # skip permission check if it's the leader
    if not _is_request_from_leader(auth_info.projects_role):
        projects_output = await run_in_threadpool(
            get_project_member().list_projects,
            db_session,
            owner,
            mlrun.api.schemas.ProjectsFormat.name_only,
            labels,
            state,
            auth_info.projects_role,
            auth_info.session,
        )
        allowed_project_names = await (
            mlrun.api.utils.auth.verifier.AuthVerifier().filter_projects_by_permissions(
                projects_output.projects,
                auth_info,
            )
        )
    return await run_in_threadpool(
        get_project_member().list_projects,
        db_session,
        owner,
        format_,
        labels,
        state,
        auth_info.projects_role,
        auth_info.session,
        allowed_project_names,
    )


@router.get(
    "/project-summaries", response_model=mlrun.api.schemas.ProjectSummariesOutput
)
async def list_project_summaries(
    owner: str = None,
    labels: typing.List[str] = fastapi.Query(None, alias="label"),
    state: mlrun.api.schemas.ProjectState = None,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    projects_output = await run_in_threadpool(
        get_project_member().list_projects,
        db_session,
        owner,
        mlrun.api.schemas.ProjectsFormat.name_only,
        labels,
        state,
        auth_info.projects_role,
        auth_info.session,
    )
    allowed_project_names = projects_output.projects
    # skip permission check if it's the leader
    if not _is_request_from_leader(auth_info.projects_role):
        allowed_project_names = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_projects_by_permissions(
            projects_output.projects,
            auth_info,
        )
    return await get_project_member().list_project_summaries(
        db_session,
        owner,
        labels,
        state,
        auth_info.projects_role,
        auth_info.session,
        allowed_project_names,
    )


@router.get(
    "/project-summaries/{name}", response_model=mlrun.api.schemas.ProjectSummary
)
async def get_project_summary(
    name: str,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    project_summary = await get_project_member().get_project_summary(
        db_session, name, auth_info.session
    )
    # skip permission check if it's the leader
    if not _is_request_from_leader(auth_info.projects_role):
        await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.api.schemas.AuthorizationAction.read,
            auth_info,
        )
    return project_summary


def _is_request_from_leader(
    projects_role: typing.Optional[mlrun.api.schemas.ProjectsRole],
) -> bool:
    if projects_role and projects_role.value == mlrun.mlconf.httpdb.projects.leader:
        return True
    return False
