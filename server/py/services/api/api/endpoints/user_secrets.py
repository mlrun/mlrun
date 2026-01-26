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
from typing import Optional

import fastapi
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.errors

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
    username: Optional[str] = fastapi.Query(
        default=None,
        description="Username to filter tokens. Use '*' to list all users' tokens (system-admin only).",
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
):
    """
    List secret tokens.

    Authorization logic:
    - Regular users:
      - None, "", or own username -> lists their own tokens
      - Any other username -> raises MLRunAccessDeniedError
    - Admin users:
      - None or "" -> lists their own tokens
      - "*" -> lists tokens for ALL users
      - Specific username -> lists that user's tokens
    """
    target_username = await _resolve_target_username_for_list_secret_tokens(
        auth_info, username
    )
    return await run_in_threadpool(
        services.api.crud.Secrets().list_secret_tokens,
        auth_info,
        target_username,
    )


@router.delete(
    "/tokens/{name}",
    status_code=HTTPStatus.OK.value,
    response_model=mlrun.common.schemas.DeleteSecretTokenResponse,
)
async def delete_secret_token(
    name: str,
    username: Optional[str] = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
):
    """
    Delete a secret token.

    Authorization logic:
    - Regular users:
      - None, "", or own username -> deletes their own token
      - Any other username -> raises MLRunAccessDeniedError
    - Admin users:
      - None or "" -> deletes their own token
      - Specific username -> deletes that user's token

    Returns:
        DeleteSecretTokenResponse with deleted=True if token was deleted,
        or deleted=False if token was not found.
    """
    target_username = await _resolve_target_username_for_delete_secret_tokens(
        auth_info, username
    )
    return await run_in_threadpool(
        services.api.crud.Secrets().delete_secret_token,
        name,
        target_username,
        auth_info,
    )


async def _resolve_target_username_for_list_secret_tokens(
    auth_info: mlrun.common.schemas.AuthInfo,
    username: Optional[str],
) -> str:
    """
    Resolve the target username for LIST token operations.

    Regular users:
      - None, "", or self -> return auth_info.user id (own tokens)
      - any other username -> raise MLRunAccessDeniedError

    Users with System-Admin permissions:
      - None or "" -> return auth_info.username (own tokens)
      - "*" -> return None (all users)
      - specific username -> return that username
    """
    # No username provided (or username="") -> return own tokens for both regular user and admin
    if not username:
        return auth_info.username

    has_system_admin_permissions = await framework.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.tokens,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
        raise_on_forbidden=False,
        resource_namespace=mlrun.common.schemas.AuthorizationResourceNamespace.mgmt,
    )

    # "*" wildcard -> system-admin only, returns all users
    if username == "*":
        if not has_system_admin_permissions:
            raise mlrun.errors.MLRunAccessDeniedError(
                "Only system admins can list tokens for all users"
            )
        return username

    # Specific username provided
    # Regular users can only query themselves
    if not has_system_admin_permissions and username != auth_info.username:
        raise mlrun.errors.MLRunAccessDeniedError(
            "Only system admins can list tokens for other users"
        )

    return username


async def _resolve_target_username_for_delete_secret_tokens(
    auth_info: mlrun.common.schemas.AuthInfo,
    username: Optional[str],
) -> str:
    """
    Resolve the target username for DELETE token operations.

    Regular users:
      - None, "", or self -> return auth_info.username (own token)
      - any other username -> raise MLRunAccessDeniedError

    Users with System-Admin permissions:
      - None or "" -> return auth_info.username (own token)
      - specific username -> return that username
    """
    # No username provided (or username="") -> delete own token for both regular user and admin
    if not username:
        return auth_info.username

    has_system_admin_permissions = await framework.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.tokens,
        action=mlrun.common.schemas.AuthorizationAction.delete,
        auth_info=auth_info,
        raise_on_forbidden=False,
        resource_namespace=mlrun.common.schemas.AuthorizationResourceNamespace.mgmt,
    )

    # Specific username provided
    # Regular users can only delete their own tokens
    if not has_system_admin_permissions and username != auth_info.username:
        raise mlrun.errors.MLRunAccessDeniedError(
            "Only system admins can delete tokens for other users"
        )

    return username
