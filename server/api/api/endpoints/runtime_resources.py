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
import copy
import http
import typing

import fastapi
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun
import mlrun.common.schemas
import server.api.api.deps
import server.api.crud
import server.api.utils.auth.verifier

router = fastapi.APIRouter(prefix="/projects/{project}/runtime-resources")


@router.get(
    "",
    response_model=typing.Union[
        mlrun.common.schemas.RuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ],
)
async def list_runtime_resources(
    project: str,
    label_selector: typing.Optional[str] = fastapi.Query(None, alias="label-selector"),
    kind: typing.Optional[str] = None,
    object_id: typing.Optional[str] = fastapi.Query(None, alias="object-id"),
    group_by: typing.Optional[
        mlrun.common.schemas.ListRuntimeResourcesGroupByField
    ] = fastapi.Query(None, alias="group-by"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    return await _list_runtime_resources(
        project, auth_info, label_selector, group_by, kind, object_id
    )


@router.delete(
    "",
    response_model=mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
)
async def delete_runtime_resources(
    project: str,
    label_selector: typing.Optional[str] = fastapi.Query(None, alias="label-selector"),
    kind: typing.Optional[str] = None,
    object_id: typing.Optional[str] = fastapi.Query(None, alias="object-id"),
    force: bool = False,
    grace_period: int = fastapi.Query(
        mlrun.mlconf.runtime_resources_deletion_grace_period, alias="grace-period"
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        server.api.api.deps.get_db_session
    ),
):
    return await _delete_runtime_resources(
        db_session,
        auth_info,
        project,
        label_selector,
        kind,
        object_id,
        force,
        grace_period,
    )


async def _delete_runtime_resources(
    db_session: sqlalchemy.orm.Session,
    auth_info: mlrun.common.schemas.AuthInfo,
    project: str,
    label_selector: typing.Optional[str] = None,
    kind: typing.Optional[str] = None,
    object_id: typing.Optional[str] = None,
    force: bool = False,
    grace_period: typing.Optional[int] = None,
    return_body: bool = True,
) -> typing.Union[
    mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput, fastapi.Response
]:
    (
        allowed_projects,
        grouped_by_project_runtime_resources_output,
        is_non_project_runtime_resource_exists,
        not_allowed_projects_exist,
    ) = await _get_runtime_resources_allowed_projects(
        project,
        auth_info,
        label_selector,
        kind,
        object_id,
        mlrun.common.schemas.AuthorizationAction.delete,
    )

    # TODO: once we have more granular permissions, we should check if the user is allowed to delete the specific
    #  runtime resources and not just the project in general
    if not_allowed_projects_exist:
        # if the user is not allowed to delete at least one of the projects, we return 403 as to:
        # 1. not leak information about the existence of not allowed projects
        # 2. not allow the user to do a partial delete action (delete some projects' resources and not others)
        raise mlrun.errors.MLRunAccessDeniedError(
            "Access denied to one or more runtime resources"
        )

    # if nothing allowed, simply return empty response
    if allowed_projects:
        permissions_label_selector = _generate_label_selector_for_allowed_projects(
            allowed_projects
        )
        if label_selector:
            computed_label_selector = ",".join(
                [label_selector, permissions_label_selector]
            )
        else:
            computed_label_selector = permissions_label_selector
        await run_in_threadpool(
            server.api.crud.RuntimeResources().delete_runtime_resources,
            db_session,
            kind,
            object_id,
            computed_label_selector,
            force,
            grace_period,
        )
    if is_non_project_runtime_resource_exists:
        # delete one more time, without adding the allowed projects selector
        await run_in_threadpool(
            server.api.crud.RuntimeResources().delete_runtime_resources,
            db_session,
            kind,
            object_id,
            label_selector,
            force,
            grace_period,
        )
    if return_body:
        filtered_projects = copy.deepcopy(allowed_projects)
        if is_non_project_runtime_resource_exists:
            filtered_projects.append("")
        return server.api.crud.RuntimeResources().filter_and_format_grouped_by_project_runtime_resources_output(
            grouped_by_project_runtime_resources_output,
            filtered_projects,
            mlrun.common.schemas.ListRuntimeResourcesGroupByField.project,
        )
    else:
        return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


async def _list_runtime_resources(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo,
    label_selector: typing.Optional[str] = None,
    group_by: typing.Optional[
        mlrun.common.schemas.ListRuntimeResourcesGroupByField
    ] = None,
    kind_filter: typing.Optional[str] = None,
    object_id: typing.Optional[str] = None,
) -> typing.Union[
    mlrun.common.schemas.RuntimeResourcesOutput,
    mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
    mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
]:
    (
        allowed_projects,
        grouped_by_project_runtime_resources_output,
        _,
        _,
    ) = await _get_runtime_resources_allowed_projects(
        project, auth_info, label_selector, kind_filter, object_id
    )
    return server.api.crud.RuntimeResources().filter_and_format_grouped_by_project_runtime_resources_output(
        grouped_by_project_runtime_resources_output,
        allowed_projects,
        group_by,
    )


async def _get_runtime_resources_allowed_projects(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo,
    label_selector: typing.Optional[str] = None,
    kind: typing.Optional[str] = None,
    object_id: typing.Optional[str] = None,
    action: mlrun.common.schemas.AuthorizationAction = mlrun.common.schemas.AuthorizationAction.read,
) -> tuple[
    list[str],
    mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    bool,
    bool,
]:
    if project != "*":
        await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    grouped_by_project_runtime_resources_output: (
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput
    )
    grouped_by_project_runtime_resources_output = await run_in_threadpool(
        server.api.crud.RuntimeResources().list_runtime_resources,
        project,
        kind,
        object_id,
        label_selector,
        mlrun.common.schemas.ListRuntimeResourcesGroupByField.project,
    )

    projects = []
    is_non_project_runtime_resource_exists = False
    for (
        project,
        kind_runtime_resources_map,
    ) in grouped_by_project_runtime_resources_output.items():
        if not project:
            is_non_project_runtime_resource_exists = True
            continue
        projects.append(project)
    allowed_projects = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.runtime_resource,
        projects,
        lambda project: (
            project,
            "",
        ),
        auth_info,
        action=action,
    )
    not_allowed_projects_exist = len(projects) != len(allowed_projects)
    return (
        allowed_projects,
        grouped_by_project_runtime_resources_output,
        is_non_project_runtime_resource_exists,
        not_allowed_projects_exist,
    )


def _generate_label_selector_for_allowed_projects(
    allowed_projects: list[str],
):
    return f"mlrun/project in ({', '.join(allowed_projects)})"
