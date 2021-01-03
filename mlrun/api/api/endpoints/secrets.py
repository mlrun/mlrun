from http import HTTPStatus
from fastapi import APIRouter, Response
from mlrun.api import schemas
from mlrun.utils.vault import (
    add_vault_user_secrets,
    add_vault_project_secrets,
    init_project_vault_configuration,
)

router = APIRouter()


@router.post("/projects/{project}/secrets")
def initialize_project_secrets(
    project: str, secrets: schemas.SecretCreationRequest,
):
    if secrets.provider != schemas.SecretProviderName.vault:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.vault,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    # Init is idempotent and will do nothing if infra is already in place
    init_project_vault_configuration(project)
    add_vault_project_secrets(project, secrets.secrets)
    return Response(status_code=HTTPStatus.CREATED.value)


@router.post("/user-secrets")
def add_user_secrets(secrets: schemas.UserSecretCreationRequest,):
    if secrets.provider != schemas.SecretProviderName.vault:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.vault,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    add_vault_user_secrets(secrets.user, secrets.secrets)
    return Response(status_code=HTTPStatus.CREATED.value)
