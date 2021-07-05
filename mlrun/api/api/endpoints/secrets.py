from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Header, Query, Response

import mlrun.errors
from mlrun.api import schemas
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.utils.vault import (
    VaultStore,
    add_vault_project_secrets,
    add_vault_user_secrets,
    init_project_vault_configuration,
)

router = APIRouter()


@router.post("/projects/{project}/secrets", status_code=HTTPStatus.CREATED.value)
def initialize_project_secrets(
    project: str, secrets: schemas.SecretsData,
):
    if secrets.provider == schemas.SecretProviderName.vault:
        # Init is idempotent and will do nothing if infra is already in place
        init_project_vault_configuration(project)

        # If no secrets were passed, no need to touch the actual secrets.
        if secrets.secrets:
            add_vault_project_secrets(project, secrets.secrets)
    elif secrets.provider == schemas.SecretProviderName.kubernetes:
        # K8s secrets is the only other option right now
        if get_k8s():
            get_k8s().store_project_secrets(project, secrets.secrets)
        else:
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Provider requested is not supported. provider = {secrets.provider}"
        )

    return Response(status_code=HTTPStatus.CREATED.value)


@router.delete("/projects/{project}/secrets", status_code=HTTPStatus.NO_CONTENT.value)
def delete_project_secrets(
    project: str, provider: str, secrets: List[str] = Query(None, alias="secret"),
):
    if provider == schemas.SecretProviderName.vault:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Delete secret is not implemented for provider {provider}"
        )
    elif provider == schemas.SecretProviderName.kubernetes:
        if get_k8s():
            get_k8s().delete_project_secrets(project, secrets)
        else:
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Provider requested is not supported. provider = {provider}"
        )

    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/secret-keys", response_model=schemas.SecretKeysData)
def list_secret_keys(
    project: str,
    provider: schemas.SecretProviderName = schemas.SecretProviderName.vault,
    token: str = Header(None, alias=schemas.HeaderNames.secret_store_token),
):
    if provider == schemas.SecretProviderName.vault:
        if not token:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Vault list project secret keys request without providing token"
            )

        vault = VaultStore(token)
        secret_values = vault.get_secrets(None, project=project)
        return schemas.SecretKeysData(
            provider=provider, secret_keys=list(secret_values.keys())
        )
    elif provider == schemas.SecretProviderName.kubernetes:
        if token:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot specify token when requesting k8s secret keys"
            )

        if get_k8s():
            secret_keys = get_k8s().get_project_secret_keys(project) or []
            return schemas.SecretKeysData(provider=provider, secret_keys=secret_keys)
        else:
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Provider requested is not supported. provider = {provider}"
        )


@router.get("/projects/{project}/secrets", response_model=schemas.SecretsData)
def list_secrets(
    project: str,
    secrets: List[str] = Query(None, alias="secret"),
    provider: schemas.SecretProviderName = schemas.SecretProviderName.vault,
    token: str = Header(None, alias=schemas.HeaderNames.secret_store_token),
):
    if provider == schemas.SecretProviderName.vault:
        if not token:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Vault list project secrets request without providing token"
            )

        vault = VaultStore(token)
        secret_values = vault.get_secrets(secrets, project=project)
        return schemas.SecretsData(provider=provider, secrets=secret_values)
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Provider requested is not supported. provider = {provider}"
        )


@router.post("/user-secrets", status_code=HTTPStatus.CREATED.value)
def add_user_secrets(secrets: schemas.UserSecretCreationRequest,):
    if secrets.provider != schemas.SecretProviderName.vault:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.vault,
            content=f"Invalid secrets provider {secrets.provider}",
        )

    add_vault_user_secrets(secrets.user, secrets.secrets)
    return Response(status_code=HTTPStatus.CREATED.value)
