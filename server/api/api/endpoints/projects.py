# Copyright 2023 Iguazio
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

import fastapi
import semver
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.formatters
import mlrun.common.schemas
import server.api.api.deps
import server.api.api.utils
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.clients.chief
import server.api.utils.helpers
from mlrun.utils import logger
from server.api.utils.singletons.project_member import get_project_member

router = fastapi.APIRouter()


@router.post(
    "/projects",
    responses={
        http.HTTPStatus.CREATED.value: {"model": mlrun.common.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def create_project(
    project: mlrun.common.schemas.Project,
    response: fastapi.Response,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
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
        http.HTTPStatus.OK.value: {"model": mlrun.common.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
async def store_project(
    project: mlrun.common.schemas.Project,
    name: str,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    project, is_running_in_background = await run_in_threadpool(
        get_project_member().store_project,
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
        http.HTTPStatus.OK.value: {"model": mlrun.common.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
def patch_project(
    project: dict,
    name: str,
    patch_mode: mlrun.common.schemas.PatchMode = fastapi.Header(
        mlrun.common.schemas.PatchMode.replace,
        alias=mlrun.common.schemas.HeaderNames.patch_mode,
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
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


@router.get("/projects/{name}", response_model=mlrun.common.schemas.ProjectOutput)
async def get_project(
    name: str,
    format_: mlrun.common.formatters.ProjectFormat = fastapi.Query(
        mlrun.common.formatters.ProjectFormat.full, alias="format"
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    project = await run_in_threadpool(
        get_project_member().get_project,
        db_session,
        name,
        auth_info.session,
        format_=format_,
    )
    # skip permission check if it's the leader
    if not server.api.utils.helpers.is_request_from_leader(auth_info.projects_role):
        await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.common.schemas.AuthorizationAction.read,
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
    background_tasks: fastapi.BackgroundTasks,
    name: str,
    request: fastapi.Request,
    deletion_strategy: mlrun.common.schemas.DeletionStrategy = fastapi.Header(
        mlrun.common.schemas.DeletionStrategy.default(),
        alias=mlrun.common.schemas.HeaderNames.deletion_strategy,
    ),
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    # check if project exists
    try:
        project = await run_in_threadpool(
            get_project_member().get_project, db_session, name, auth_info.session
        )
    except mlrun.errors.MLRunNotFoundError:
        logger.info("Project not found, nothing to delete", project=name)
        return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)

    # delete project can be responsible for deleting schedules. Schedules are running only on chief,
    # that is why we re-route requests to chief
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to delete project, re-routing to chief",
            project=name,
            deletion_strategy=deletion_strategy,
        )
        chief_client = server.api.utils.clients.chief.Client()
        return await chief_client.delete_project(name=name, request=request)

    # we need to implement the verify_project_is_empty, since we don't want
    # to spawn a background task for this, only to return a response
    if deletion_strategy.strategy_to_check():
        server.api.crud.Projects().verify_project_is_empty(db_session, name, auth_info)
        if deletion_strategy == mlrun.common.schemas.DeletionStrategy.check:
            # if the strategy is check, we don't want to delete the project, only to check if it is empty
            return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)
        elif deletion_strategy.is_restricted():
            # if the deletion strategy is restricted, and we passed validation, we want to go through the deletion
            # process even if resources are created in the project after this point (for example in
            # process_model_monitoring_secret).
            # therefore, we change the deletion strategy to cascading to both ensure we won't fail later, and that we
            # will delete the project and all its resources.
            deletion_strategy = mlrun.common.schemas.DeletionStrategy.cascading

    igz_version = mlrun.mlconf.get_parsed_igz_version()
    if (
        server.api.utils.helpers.is_request_from_leader(auth_info.projects_role)
        and igz_version
        and igz_version < semver.VersionInfo.parse("3.5.5")
    ):
        # here in DELETE v1/projects, if the leader is iguazio < 3.5.5, the leader isn't waiting for the background
        # task from v2 to complete. In order for this request not to time out, we want to start the background task
        # for deleting the project and return 202 to the leader. Later, in the project deletion wrapper task, we will
        # wait for this background task to complete before marking the task as done.
        task, _ = await run_in_threadpool(
            server.api.api.utils.get_or_create_project_deletion_background_task,
            project,
            deletion_strategy,
            db_session,
            auth_info,
        )
        if task:
            background_tasks.add_task(task)
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)

    is_running_in_background = False
    force_delete = False
    try:
        is_running_in_background = await run_in_threadpool(
            get_project_member().delete_project,
            db_session,
            name,
            deletion_strategy,
            auth_info.projects_role,
            auth_info,
            wait_for_completion=wait_for_completion,
        )
    except mlrun.errors.MLRunNotFoundError as exc:
        if server.api.utils.helpers.is_request_from_leader(auth_info.projects_role):
            raise exc

        if project.status.state != mlrun.common.schemas.ProjectState.archived:
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Failed to delete project {name}. Project not found in leader, but it is not in archived state."
            )

        logger.warning(
            "Project not found in leader, ensuring project deleted in mlrun",
            project_name=name,
            err=mlrun.errors.err_to_str(exc),
        )
        force_delete = True

    if force_delete:
        # In this case the wrapper delete project request is the one deleting the project because it
        # doesn't exist in the leader.
        await run_in_threadpool(
            server.api.crud.Projects().delete_project,
            db_session,
            name,
            deletion_strategy,
            auth_info,
        )

    elif is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)

    else:
        # For iguazio < 3.5.5, the project deletion job is triggered while iguazio does not wait for it to complete.
        # We wait for it here to make sure we respond with a proper status code.
        await run_in_threadpool(
            server.api.api.utils.verify_project_is_deleted, name, auth_info
        )

    await get_project_member().post_delete_project(name)
    if force_delete:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


@router.get("/projects", response_model=mlrun.common.schemas.ProjectsOutput)
async def list_projects(
    format_: mlrun.common.formatters.ProjectFormat = fastapi.Query(
        mlrun.common.formatters.ProjectFormat.full, alias="format"
    ),
    owner: str = None,
    labels: list[str] = fastapi.Query(None, alias="label"),
    state: mlrun.common.schemas.ProjectState = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    allowed_project_names = None
    # skip permission check if it's the leader
    if not server.api.utils.helpers.is_request_from_leader(auth_info.projects_role):
        projects_output = await run_in_threadpool(
            get_project_member().list_projects,
            db_session,
            owner,
            mlrun.common.formatters.ProjectFormat.name_only,
            labels,
            state,
            auth_info.projects_role,
            auth_info.session,
        )
        allowed_project_names = await server.api.utils.auth.verifier.AuthVerifier().filter_projects_by_permissions(
            projects_output.projects,
            auth_info,
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
    "/project-summaries", response_model=mlrun.common.schemas.ProjectSummariesOutput
)
async def list_project_summaries(
    owner: str = None,
    labels: list[str] = fastapi.Query(None, alias="label"),
    state: mlrun.common.schemas.ProjectState = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    projects_output = await run_in_threadpool(
        get_project_member().list_projects,
        db_session,
        owner,
        mlrun.common.formatters.ProjectFormat.name_only,
        labels,
        state,
        auth_info.projects_role,
        auth_info.session,
    )
    allowed_project_names = projects_output.projects
    # skip permission check if it's the leader
    if not server.api.utils.helpers.is_request_from_leader(auth_info.projects_role):
        auth_verifier = server.api.utils.auth.verifier.AuthVerifier()
        allowed_project_names = await auth_verifier.filter_project_resources_by_permissions(
            resource_type=mlrun.common.schemas.AuthorizationResourceTypes.project_summaries,
            resources=allowed_project_names,
            project_and_resource_name_extractor=lambda project: (
                project,
                "",
            ),
            auth_info=auth_info,
            action=mlrun.common.schemas.AuthorizationAction.read,
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
    "/project-summaries/{name}", response_model=mlrun.common.schemas.ProjectSummary
)
async def get_project_summary(
    name: str,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    project_summary = await get_project_member().get_project_summary(
        db_session, name, auth_info.session
    )
    # skip permission check if it's the leader
    if not server.api.utils.helpers.is_request_from_leader(auth_info.projects_role):
        await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    return project_summary


@router.post("/projects/{name}/load")
async def load_project(
    name: str,
    url: str,
    secrets: mlrun.common.schemas.SecretsData = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    """
    Loading a project remotely from a given source.

    :param name:                project name
    :param url:                 git or tar.gz or .zip sources archive path e.g.:
                                git://github.com/mlrun/demo-xgb-project.git
                                http://mysite/archived-project.zip
                                The git project should include the project yaml file.
    :param secrets:             Secrets to store in project in order to load it from the provided url.
                                For more information see :py:func:`mlrun.load_project` function.
    :param auth_info:           auth info of the request
    :param db_session:          session that manages the current dialog with the database

    :returns: a Run object of the load project function
    """

    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name),
        spec=mlrun.common.schemas.ProjectSpec(source=url),
    )

    # We must create the project before we run the remote load_project function because
    # we want this function will be running under the project itself instead of the default project.
    project, _ = await fastapi.concurrency.run_in_threadpool(
        get_project_member().create_project,
        db_session=db_session,
        project=project,
        projects_role=auth_info.projects_role,
        leader_session=auth_info.session,
    )

    # Storing secrets in project
    if secrets is not None:
        await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.secret,
            project.metadata.name,
            secrets.provider,
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )

        await run_in_threadpool(
            server.api.crud.Secrets().store_project_secrets,
            project.metadata.name,
            secrets,
        )

    # Creating the auxiliary function for loading the project:
    load_project_runner = await fastapi.concurrency.run_in_threadpool(
        server.api.crud.WorkflowRunners().create_runner,
        run_name=f"load-{name}",
        project=name,
        db_session=db_session,
        auth_info=auth_info,
        image=mlrun.mlconf.default_base_image,
    )

    logger.debug(
        "Saved function for loading project",
        project_name=name,
        function_name=load_project_runner.metadata.name,
        kind=load_project_runner.kind,
        source=project.spec.source,
    )

    run = await fastapi.concurrency.run_in_threadpool(
        server.api.crud.WorkflowRunners().run,
        runner=load_project_runner,
        project=project,
        workflow_request=None,
        load_only=True,
    )
    return {"data": run.to_dict()}
