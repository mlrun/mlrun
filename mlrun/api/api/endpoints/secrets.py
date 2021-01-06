from http import HTTPStatus
from fastapi import APIRouter, Response, Header, Query
from typing import List
from mlrun.api import schemas
from mlrun.utils.vault import (
    VaultStore,
    add_vault_user_secrets,
    add_vault_project_secrets,
    init_project_vault_configuration,
)

router = APIRouter()


@router.post("/projects/{project}/secrets", status_code=HTTPStatus.CREATED.value)
def initialize_project_secrets(
    project: str, secrets: schemas.SecretsData,
):
    if secrets.provider != schemas.SecretProviderName.vault:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    # Init is idempotent and will do nothing if infra is already in place
    init_project_vault_configuration(project)
    add_vault_project_secrets(project, secrets.secrets)
    return Response(status_code=HTTPStatus.CREATED.value)


@router.get("/projects/{project}/secrets", response_model=schemas.SecretsData)
def get_secrets(
    project: str,
    secrets: List[str] = Query(None, alias="secret"),
    provider: schemas.SecretProviderName = schemas.SecretProviderName.vault,
    token: str = Header(None, alias=schemas.HeaderNames.secret_store_token),
):
    if not token:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            content="Get project secrets request without providing token",
        )
    if provider != schemas.SecretProviderName.vault:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    vault = VaultStore(token)
    secret_values = vault.get_secrets(secrets, project=project)
    return schemas.SecretsData(provider=provider, secrets=secret_values)


@router.post("/user-secrets", status_code=HTTPStatus.CREATED.value)
def add_user_secrets(secrets: schemas.UserSecretCreationRequest,):
    if secrets.provider != schemas.SecretProviderName.vault:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.vault,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    add_vault_user_secrets(secrets.user, secrets.secrets)
    return Response(status_code=HTTPStatus.CREATED.value)
