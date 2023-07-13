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
from http import HTTPStatus
from typing import List

import fastapi
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.common.schemas
import mlrun.errors

router = fastapi.APIRouter()


@router.post("/projects/{project}/secrets", status_code=HTTPStatus.CREATED.value)
async def store_project_secrets(
    project: str,
    secrets: mlrun.common.schemas.SecretsData,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    # Doing a specific check for project existence, because we want to return 404 in the case of a project not
    # existing, rather than returning a permission error, as it misleads the user. We don't even care for return
    # value.
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project,
        auth_info.session,
    )

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project,
        secrets.provider,
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Secrets().store_project_secrets, project, secrets
    )

    return fastapi.Response(status_code=HTTPStatus.CREATED.value)


@router.delete("/projects/{project}/secrets", status_code=HTTPStatus.NO_CONTENT.value)
async def delete_project_secrets(
    project: str,
    provider: mlrun.common.schemas.SecretProviderName = mlrun.common.schemas.SecretProviderName.kubernetes,
    secrets: List[str] = fastapi.Query(None, alias="secret"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project,
        auth_info.session,
    )

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project,
        provider,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await run_in_threadpool(
        mlrun.api.crud.Secrets().delete_project_secrets, project, provider, secrets
    )

    return fastapi.Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get(
    "/projects/{project}/secret-keys",
    response_model=mlrun.common.schemas.SecretKeysData,
)
async def list_project_secret_keys(
    project: str,
    provider: mlrun.common.schemas.SecretProviderName = mlrun.common.schemas.SecretProviderName.kubernetes,
    token: str = fastapi.Header(
        None, alias=mlrun.common.schemas.HeaderNames.secret_store_token
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project,
        auth_info.session,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project,
        provider,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return await run_in_threadpool(
        mlrun.api.crud.Secrets().list_project_secret_keys, project, provider, token
    )


@router.get(
    "/projects/{project}/secrets", response_model=mlrun.common.schemas.SecretsData
)
async def list_project_secrets(
    project: str,
    secrets: List[str] = fastapi.Query(None, alias="secret"),
    provider: mlrun.common.schemas.SecretProviderName = mlrun.common.schemas.SecretProviderName.kubernetes,
    token: str = fastapi.Header(
        None, alias=mlrun.common.schemas.HeaderNames.secret_store_token
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(mlrun.api.api.deps.get_db_session),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project,
        auth_info.session,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project,
        provider,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return await run_in_threadpool(
        mlrun.api.crud.Secrets().list_project_secrets, project, provider, secrets, token
    )


@router.post("/user-secrets", status_code=HTTPStatus.CREATED.value)
def add_user_secrets(
    secrets: mlrun.common.schemas.UserSecretCreationRequest,
):
    # vault is not used
    return fastapi.Response(
        status_code=HTTPStatus.BAD_REQUEST.value,
        content=f"Invalid secrets provider {secrets.provider}",
    )
