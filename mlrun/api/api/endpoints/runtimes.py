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


@router.get("/runtimes")
def list_runtimes(
    label_selector: str = None,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    project = "*"
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.runtime_resource,
        project,
        "",
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    return mlrun.api.crud.Runtimes().list_runtimes(project, label_selector)


# TODO: move everything to use this endpoint instead of list_runtimes and deprecate it
@router.get("/projects/{project}/runtime-resources")
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
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.runtime_resource,
        project,
        "",
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    return mlrun.api.crud.Runtimes().list_runtimes(project, label_selector, group_by)


@router.get("/runtimes/{kind}")
def get_runtime(kind: str, label_selector: str = None):
    return mlrun.api.crud.Runtimes().get_runtime(kind, label_selector)


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
