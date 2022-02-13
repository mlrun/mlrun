from http import HTTPStatus
from typing import List

import fastapi
from sqlalchemy.orm import Session

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.errors
from mlrun.api import schemas
from mlrun.utils.vault import add_vault_user_secrets

router = fastapi.APIRouter()


@router.post("/projects/{project}/secrets", status_code=HTTPStatus.CREATED.value)
def store_project_secrets(
    project: str,
    secrets: schemas.SecretsData,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    # Doing a specific check for project existence, because we want to return 404 in the case of a project not
    # existing, rather than returning a permission error, as it misleads the user. We don't even care for return
    # value.
    mlrun.api.utils.singletons.project_member.get_project_member().get_project(
        db_session, project, auth_info.session
    )

    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.secret,
        project,
        secrets.provider,
        mlrun.api.schemas.AuthorizationAction.create,
        auth_info,
    )
    mlrun.api.crud.Secrets().store_secrets(project, secrets)

    return fastapi.Response(status_code=HTTPStatus.CREATED.value)


@router.delete("/projects/{project}/secrets", status_code=HTTPStatus.NO_CONTENT.value)
def delete_project_secrets(
    project: str,
    provider: schemas.SecretProviderName = schemas.SecretProviderName.kubernetes,
    secrets: List[str] = fastapi.Query(None, alias="secret"),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    mlrun.api.utils.singletons.project_member.get_project_member().get_project(
        db_session, project, auth_info.session
    )

    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.secret,
        project,
        provider,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_info,
    )
    mlrun.api.crud.Secrets().delete_secrets(project, provider, secrets)

    return fastapi.Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/secret-keys", response_model=schemas.SecretKeysData)
def list_secret_keys(
    project: str,
    provider: schemas.SecretProviderName = schemas.SecretProviderName.kubernetes,
    token: str = fastapi.Header(None, alias=schemas.HeaderNames.secret_store_token),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    mlrun.api.utils.singletons.project_member.get_project_member().get_project(
        db_session, project, auth_info.session
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.secret,
        project,
        provider,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    return mlrun.api.crud.Secrets().list_secret_keys(project, provider, token)


@router.get("/projects/{project}/secrets", response_model=schemas.SecretsData)
def list_secrets(
    project: str,
    secrets: List[str] = fastapi.Query(None, alias="secret"),
    provider: schemas.SecretProviderName = schemas.SecretProviderName.kubernetes,
    token: str = fastapi.Header(None, alias=schemas.HeaderNames.secret_store_token),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    mlrun.api.utils.singletons.project_member.get_project_member().get_project(
        db_session, project, auth_info.session
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.secret,
        project,
        provider,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    return mlrun.api.crud.Secrets().list_secrets(project, provider, secrets, token)


@router.post("/user-secrets", status_code=HTTPStatus.CREATED.value)
def add_user_secrets(secrets: schemas.UserSecretCreationRequest,):
    if secrets.provider != schemas.SecretProviderName.vault:
        return fastapi.Response(
            status_code=HTTPStatus.BAD_REQUEST.vault,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    add_vault_user_secrets(secrets.user, secrets.secrets)
    return fastapi.Response(status_code=HTTPStatus.CREATED.value)
