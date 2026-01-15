# Copyright 2025 Iguazio
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

import fastapi
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas

import framework.api.deps
import framework.utils.auth.verifier
import services.api.crud

router = fastapi.APIRouter(prefix="/user-secrets")


@router.put(
    "/tokens",
    status_code=HTTPStatus.OK.value,
    response_model=mlrun.common.schemas.StoreSecretTokensResponse,
)
async def store_secret_tokens(
    secret_tokens: list[mlrun.common.schemas.SecretToken],
    force: bool = False,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
):
    return await run_in_threadpool(
        services.api.crud.Secrets().store_secret_tokens,
        secret_tokens,
        auth_info,
        force,
    )


@router.get("/tokens", response_model=mlrun.common.schemas.ListSecretTokensResponse)
async def list_secret_tokens(
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
):
    # TODO: Support listing user tokens with System Admin (ML-10775)

    return await run_in_threadpool(
        services.api.crud.Secrets().list_secret_tokens,
        auth_info,
    )


@router.delete("/tokens/{name}", status_code=HTTPStatus.OK.value)
async def revoke_secret_token(
    name: str,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
):
    # TODO: Support revoking user token with System Admin (ML-10775)

    return await run_in_threadpool(
        services.api.crud.Secrets().revoke_secret_token,
        name,
        auth_info,
    )
