import http
import typing

import fastapi
import sqlalchemy.orm

import mlrun
import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.opa

router = fastapi.APIRouter()


@router.get("/runtimes", response_model=mlrun.api.schemas.RuntimeResourcesOutput)
# TODO: remove when 0.6.6 is no longer relevant
def list_runtime_resources_legacy(
    label_selector: str = None,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    _list_runtime_resources("*", auth_verifier.auth_info, label_selector)


@router.get(
    "/projects/{project}/runtime-resources",
    response_model=typing.Union[
        mlrun.api.schemas.RuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
    ],
)
def list_runtime_resources(
    project: str,
    label_selector: str = None,
    group_by: typing.Optional[
        mlrun.api.schemas.ListRuntimeResourcesGroupByField
    ] = fastapi.Query(None, alias="group-by"),
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    return _list_runtime_resources(
        project, auth_verifier.auth_info, label_selector, group_by
    )


@router.get("/runtimes/{kind}", response_model=mlrun.api.schemas.KindRuntimeResources)
# TODO: remove when 0.6.6 is no longer relevant
def list_kind_runtime_resources_legacy(
    kind: str,
    label_selector: str = None,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    return _list_kind_runtime_resources(
        auth_verifier.auth_info, "*", kind, label_selector
    )


@router.get(
    "/projects/{project}/runtime-resources/{kind}",
    response_model=mlrun.api.schemas.KindRuntimeResources,
)
def list_kind_runtime_resources(
    project: str,
    kind: str,
    label_selector: str = None,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    return _list_kind_runtime_resources(
        auth_verifier.auth_info, project, kind, label_selector
    )


@router.delete("/runtimes", status_code=http.HTTPStatus.NO_CONTENT.value)
def delete_runtimes(
    label_selector: str = None,
    force: bool = False,
    grace_period: int = mlrun.mlconf.runtime_resources_deletion_grace_period,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    mlrun.api.crud.Runtimes().delete_runtimes(
        db_session, label_selector, force, grace_period, auth_verifier.auth_info.session
    )
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


@router.delete("/runtimes/{kind}", status_code=http.HTTPStatus.NO_CONTENT.value)
def delete_runtime(
    kind: str,
    label_selector: str = None,
    force: bool = False,
    grace_period: int = mlrun.mlconf.runtime_resources_deletion_grace_period,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    mlrun.api.crud.Runtimes().delete_runtime(
        db_session,
        kind,
        label_selector,
        force,
        grace_period,
        auth_verifier.auth_info.session,
    )
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


# FIXME: find a more REST-y path
@router.delete(
    "/runtimes/{kind}/{object_id}", status_code=http.HTTPStatus.NO_CONTENT.value
)
def delete_runtime_object(
    kind: str,
    object_id: str,
    label_selector: str = None,
    force: bool = False,
    grace_period: int = mlrun.mlconf.runtime_resources_deletion_grace_period,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    mlrun.api.crud.Runtimes().delete_runtime_object(
        db_session,
        kind,
        object_id,
        label_selector,
        force,
        grace_period,
        auth_verifier.auth_info.session,
    )
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


def _list_runtime_resources(
    project: str,
    auth_info: mlrun.api.schemas.AuthInfo,
    label_selector: str = None,
    group_by: typing.Optional[
        mlrun.api.schemas.ListRuntimeResourcesGroupByField
    ] = None,
    kind_filter: str = None,
) -> typing.Union[
    mlrun.api.schemas.RuntimeResourcesOutput,
    mlrun.api.schemas.GroupedByJobRuntimeResourcesOutput,
    mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
]:
    grouped_by_project_runtime_resources_output: mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput
    grouped_by_project_runtime_resources_output = mlrun.api.crud.Runtimes().list_runtimes(
        project,
        label_selector,
        mlrun.api.schemas.ListRuntimeResourcesGroupByField.project,
    )
    project_and_kind_tuples = []
    for (
        project,
        kind_runtime_resources_map,
    ) in grouped_by_project_runtime_resources_output.items():
        for kind in kind_runtime_resources_map.keys():
            if not kind_filter or (kind_filter and kind_filter == kind):
                project_and_kind_tuples.append((project, kind))
    allowed_project_and_kinds = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.runtime_resource,
        project_and_kind_tuples,
        lambda project_and_kind_tuple: (
            project_and_kind_tuple[0],
            project_and_kind_tuple[1],
        ),
        auth_info,
    )
    allowed_project_to_kind_map = {}
    for project, kind in allowed_project_and_kinds:
        allowed_project_to_kind_map.setdefault(project, []).append(kind)
    return mlrun.api.crud.Runtimes().filter_and_format_grouped_by_project_runtime_resources_output(
        grouped_by_project_runtime_resources_output,
        allowed_project_to_kind_map,
        group_by,
    )


def _list_kind_runtime_resources(
    auth_info: mlrun.api.schemas.AuthInfo,
    project: str,
    kind: str,
    label_selector: str = None,
) -> mlrun.api.schemas.KindRuntimeResources:
    runtime_resources_output = _list_runtime_resources(
        project, auth_info, label_selector, kind_filter=kind
    )
    if runtime_resources_output:
        return runtime_resources_output[0]
    else:
        return mlrun.api.schemas.KindRuntimeResources(
            kind=kind, resources=mlrun.api.schemas.RuntimeResources()
        )
