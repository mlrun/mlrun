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

router = fastapi.APIRouter(prefix="/user-secrets/tokens")


@router.put("", status_code=HTTPStatus.OK.value)
async def store_secret_tokens(
    secret_tokens: list[mlrun.common.schemas.SecretToken],
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
):
    # Ensure required authentication context exists
    if not auth_info.user_id or not auth_info.username:
        raise mlrun.errors.MLRunUnauthorizedError(
            "Missing required authentication details (user_id or username)"
        )

    # TODO: add user-secrets to orca resources?
    await (
        framework.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.user_secrets,
            mlrun.common.schemas.AuthorizationAction.store,
            auth_info,
        )
    )

    # TODO we need to define MLRUN_NAMESPACE?
    await run_in_threadpool(
        services.api.crud.Secrets().store_secret_tokens,
        secret_tokens,
        auth_info.user_id,
        auth_info.username,
    )

    return fastapi.Response(status_code=HTTPStatus.OK.value)
