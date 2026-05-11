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

import datetime
import http

import fastapi
import semver
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.common.formatters
import mlrun.common.schemas
from mlrun.utils import logger

import framework.api.deps
import framework.api.utils
import framework.utils.auth.verifier
import framework.utils.clients.chief
import framework.utils.helpers
import services.api.crud
from framework.utils.singletons.project_member import get_project_member

router = fastapi.APIRouter()


@router.post(
    "/projects",
    responses={
        http.HTTPStatus.CREATED.value: {"model": mlrun.common.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
async def create_project(
    project: mlrun.common.schemas.Project,
    request: fastapi.Request,
    response: fastapi.Response,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    if mlrun.mlconf.is_iguazio_v4_mode():
        await framework.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.project_global,
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )

    # we reroute to chief to ensure project sync is handled properly
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info("Requesting to create project, re-routing to chief")
        chief_client = framework.utils.clients.chief.Client()
        return await chief_client.create_project(request=request)

    project, is_running_in_background = await run_in_threadpool(
        get_project_member().create_project,
        db_session,
        project,
        auth_info,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)

    await framework.utils.auth.verifier.AuthVerifier().ensure_project_permissions(
        project.metadata.name,
        auth_info,
    )

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
    request: fastapi.Request,
    # TODO: we're in a http request context here, therefore it doesn't make sense that by default it will hold the
    #  request until the process will be completed - after UI supports waiting - change default to False
    wait_for_completion: bool = fastapi.Query(True, alias="wait-for-completion"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    await _ensure_project_create_or_update_permissions(db_session, name, auth_info)

    # we reroute to chief to ensure project sync is handled properly
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to store project, re-routing to chief",
            project=name,
        )
        chief_client = framework.utils.clients.chief.Client()
        return await chief_client.store_project(name=name, request=request)

    project, is_running_in_background = await run_in_threadpool(
        get_project_member().store_project,
        db_session,
        name,
        project,
        auth_info,
        wait_for_completion=wait_for_completion,
    )
    if is_running_in_background:
        return fastapi.Response(status_code=http.HTTPStatus.ACCEPTED.value)

    await framework.utils.auth.verifier.AuthVerifier().ensure_project_permissions(
        project.metadata.name,
        auth_info,
    )

    return project


@router.patch(
    "/projects/{name}",
    responses={
        http.HTTPStatus.OK.value: {"model": mlrun.common.schemas.Project},
        http.HTTPStatus.ACCEPTED.value: {},
    },
)
async def patch_project(
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
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    # In IG4 mode, apply fine-grained permission checks
    # In IG3 mode, skip the check when the request comes from the leader; otherwise fall back to the standard
    # project-update permission to preserve backward compatibility.
    if mlrun.mlconf.is_iguazio_v4_mode():
        await _verify_patch_project_permissions(name, project, auth_info)
    elif not framework.utils.helpers.is_request_from_leader(auth_info.projects_role):
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.common.schemas.AuthorizationAction.update,
            auth_info,
        )
    project, is_running_in_background = await run_in_threadpool(
        get_project_member().patch_project,
        db_session,
        name,
        project,
        patch_mode,
        auth_info,
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
        framework.api.deps.get_db_session
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
):
    project = await run_in_threadpool(
        get_project_member().get_project,
        db_session,
        name,
        auth_info,
        format_=format_,
    )
    # skip permission check if it's the leader in iguazio v3 mode
    if (
        mlrun.mlconf.is_iguazio_v4_mode()
        or not framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
    ):
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
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
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    # check if project exists
    try:
        project = await run_in_threadpool(
            get_project_member().get_project, db_session, name, auth_info
        )
    except mlrun.errors.MLRunNotFoundError:
        logger.info("Project not found, nothing to delete", project=name)
        return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)

    # skip permission check if it's the leader in iguazio v3 mode
    if (
        mlrun.mlconf.is_iguazio_v4_mode()
        or not framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
    ):
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
            name,
            mlrun.common.schemas.AuthorizationAction.delete,
            auth_info,
        )

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
        chief_client = framework.utils.clients.chief.Client()
        return await chief_client.delete_project(name=name, request=request)

    # we need to implement the verify_project_is_empty, since we don't want
    # to spawn a background task for this, only to return a response
    if deletion_strategy.strategy_to_check():
        services.api.crud.Projects().verify_project_is_empty(
            db_session, name, auth_info
        )
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
        framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
        and igz_version
        and igz_version < semver.VersionInfo.parse("3.5.5")
    ):
        # here in DELETE v1/projects, if the leader is iguazio < 3.5.5, the leader isn't waiting for the background
        # task from v2 to complete. In order for this request not to time out, we want to start the background task
        # for deleting the project and return 202 to the leader. Later, in the project deletion wrapper task, we will
        # wait for this background task to complete before marking the task as done.
        task, _ = await run_in_threadpool(
            framework.api.utils.get_or_create_project_deletion_background_task,
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
            auth_info,
            wait_for_completion=wait_for_completion,
        )
    except mlrun.errors.MLRunNotFoundError as exc:
        if framework.utils.helpers.is_request_from_leader(auth_info.projects_role):
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
            services.api.crud.Projects().delete_project,
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
            framework.api.utils.verify_project_is_deleted, name, auth_info
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
    owner: str | None = None,
    labels: list[str] = fastapi.Query(None, alias="label"),
    state: mlrun.common.schemas.ProjectState = None,
    updated_after: datetime.datetime | None = fastapi.Query(
        None, alias="updated_after"
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    allowed_project_names = None
    # skip permission check if it's the leader in iguazio v3 mode
    if (
        mlrun.mlconf.is_iguazio_v4_mode()
        or not framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
    ):
        projects_output = await run_in_threadpool(
            get_project_member().list_projects,
            db_session,
            auth_info,
            owner,
            mlrun.common.formatters.ProjectFormat.name_only,
            labels,
            state,
            None,
            updated_after,
        )
        allowed_project_names = await framework.utils.auth.verifier.AuthVerifier().filter_projects_by_permissions(
            projects_output.projects,
            auth_info,
        )
    return await run_in_threadpool(
        get_project_member().list_projects,
        db_session,
        auth_info,
        owner,
        format_,
        labels,
        state,
        allowed_project_names,
        updated_after,
    )


@router.get(
    "/project-summaries", response_model=mlrun.common.schemas.ProjectSummariesOutput
)
async def list_project_summaries(
    owner: str | None = None,
    labels: list[str] = fastapi.Query(None, alias="label"),
    state: mlrun.common.schemas.ProjectState = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
):
    projects_output = await run_in_threadpool(
        get_project_member().list_projects,
        db_session,
        auth_info,
        owner,
        mlrun.common.formatters.ProjectFormat.name_only,
        labels,
        state,
    )
    allowed_project_names = projects_output.projects
    # skip permission check if it's the leader in iguazio v3 mode
    if (
        mlrun.mlconf.is_iguazio_v4_mode()
        or not framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
    ):
        allowed_project_names = await framework.utils.auth.verifier.AuthVerifier().filter_projects_by_permissions(
            allowed_project_names,
            auth_info,
        )
    return await get_project_member().list_project_summaries(
        db_session,
        auth_info,
        owner,
        labels,
        state,
        allowed_project_names,
    )


@router.get(
    "/project-summaries/{name}", response_model=mlrun.common.schemas.ProjectSummary
)
async def get_project_summary(
    name: str,
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
):
    project_summary = await get_project_member().get_project_summary(
        db_session, name, auth_info
    )
    # skip permission check if it's the leader in iguazio v3 mode
    if (
        mlrun.mlconf.is_iguazio_v4_mode()
        or not framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
    ):
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
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
        framework.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        framework.api.deps.get_db_session
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

    await _ensure_project_create_or_update_permissions(db_session, name, auth_info)

    # Ensure the project exists before calling the remote load_project function
    project, _ = await fastapi.concurrency.run_in_threadpool(
        get_project_member().create_project,
        db_session=db_session,
        project=project,
        auth_info=auth_info,
    )

    # Storing secrets in project
    if secrets is not None:
        await framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.secret,
            project.metadata.name,
            secrets.provider,
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )

        await run_in_threadpool(
            services.api.crud.Secrets().store_project_secrets,
            project.metadata.name,
            secrets,
        )

    # Creating the auxiliary function for loading the project:
    load_project_runner = await fastapi.concurrency.run_in_threadpool(
        services.api.crud.LoadRunner().create_runner,
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
        services.api.crud.LoadRunner().run,
        runner=load_project_runner,
        project=project,
    )
    return {"data": run.to_dict()}


async def _verify_patch_project_permissions(
    project_name: str,
    project_patch: dict,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    """Apply fine-grained permission checks for project patch requests.

    If the patch modifies spec.owner, verify the caller has management-level owner-update permission
      (/mgmt/projects/{project}/owner – update).
    If the patch modifies any other project properties, verify the caller has the regular resource-level
    project-update permission
      (/resources/projects/{project} – update).
    If both kinds of changes are present, both checks must pass.
    """
    auth_verifier = framework.utils.auth.verifier.AuthVerifier()
    modifies_owner = _patch_modifies_owner(project_patch)
    modifies_other = _patch_modifies_non_owner_fields(project_patch)

    if modifies_owner:
        await auth_verifier.query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.project_owner,
            project_name,
            "",
            mlrun.common.schemas.AuthorizationAction.update,
            auth_info,
            resource_namespace=mlrun.common.schemas.AuthorizationResourceNamespace.mgmt,
        )

    # Run the regular resource-level check when non-owner fields are touched, or as a fallback when the patch doesn't
    # modify the owner either (e.g. empty body), so that no request bypasses permission verification entirely.
    if modifies_other or not modifies_owner:
        await auth_verifier.query_project_permissions(
            project_name,
            mlrun.common.schemas.AuthorizationAction.update,
            auth_info,
        )


def _patch_modifies_owner(project_patch: dict) -> bool:
    """Check whether the patch dict modifies spec.owner."""
    return "owner" in project_patch.get("spec", {})


def _patch_modifies_non_owner_fields(project_patch: dict) -> bool:
    """Check whether the patch dict contains fields beyond spec.owner."""
    # Any top-level key other than "spec" is a non-owner modification.
    for key in project_patch:
        if key != "spec":
            return True
    # Only "spec" at top level — check whether spec itself contains keys beyond "owner"
    spec = project_patch.get("spec", {})
    return any(key != "owner" for key in spec)


async def _ensure_project_create_or_update_permissions(
    db_session: sqlalchemy.orm.Session,
    project_name: str,
    auth_info: mlrun.common.schemas.AuthInfo,
):
    """Ensure create or update permissions based on project existence."""
    # Only check leader header in iguazio v3
    if (
        not mlrun.mlconf.is_iguazio_v4_mode()
        and framework.utils.helpers.is_request_from_leader(auth_info.projects_role)
    ):
        return

    try:
        await run_in_threadpool(
            get_project_member().get_project,
            db_session,
            project_name,
            auth_info,
            format_=mlrun.common.formatters.ProjectFormat.name_only,
        )
        project_exists = True
    except mlrun.errors.MLRunNotFoundError:
        project_exists = False

    if project_exists:
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project_name, mlrun.common.schemas.AuthorizationAction.update, auth_info
        )
        return

    # In Iguazio v4 mode, mlrun is the project leader and main entrypoint so we must ensure
    # that the user has create permissions for projects.
    if mlrun.mlconf.is_iguazio_v4_mode():
        await framework.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.project_global,
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )
