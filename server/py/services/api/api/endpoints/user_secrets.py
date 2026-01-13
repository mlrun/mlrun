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
        username=target_username,
    )


@router.delete(
    "/tokens/{name}",
    status_code=HTTPStatus.OK.value,
    response_model=mlrun.common.schemas.RevokeSecretTokenResponse,
)
async def revoke_secret_token(
    name: str,
    username: Optional[str] = None,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
):
    """
    Revoke a secret token.

    Authorization logic:
    - Regular users:
      - None, "", or own username -> revokes their own token
      - Any other username -> raises MLRunAccessDeniedError
    - Admin users:
      - None or "" -> revokes their own token
      - Specific username -> revokes that user's token

    Returns:
        RevokeSecretTokenResponse with revoked=True if token was revoked,
        or revoked=False if token was not found.
    """
    target_username = await _resolve_target_username_for_revoke_secret_tokens(
        auth_info, username
    )
    return await run_in_threadpool(
        services.api.crud.Secrets().revoke_secret_token,
        name,
        target_username,
        auth_info.request_headers,
    )


async def _is_system_admin(
    auth_info: mlrun.common.schemas.AuthInfo,
    action: mlrun.common.schemas.AuthorizationAction,
) -> bool:
    """
    Check if the authenticated user has system admin privileges for token operations.

    System admin privileges are determined by querying the authorization provider
    for permissions on the 'tokens' resource in the 'mgmt' namespace.

    :param auth_info: Authentication information of the user.
    :param action: The authorization action to check (read, delete, etc.).
    :return: True if the user has system admin privileges, False otherwise.
    """
    return (
        await framework.utils.auth.verifier.AuthVerifier().query_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.tokens,
            "",
            action,
            auth_info,
            raise_on_forbidden=False,
            resource_namespace=mlrun.common.schemas.AuthorizationResourceNamespace.mgmt,
        )
    )


async def _resolve_target_username_for_list_secret_tokens(
    auth_info: mlrun.common.schemas.AuthInfo,
    username: Optional[str],
) -> str:
    """
    Resolve the target username for LIST token operations.

    Regular users:
      - None, "", or self -> return auth_info.username (own tokens)
      - any other username -> raise MLRunAccessDeniedError

    Admin users:
      - None or "" -> return auth_info.username (own tokens)
      - "*" -> return None (all users)
      - specific username -> return that username
    """
    # No username provided (or username="") -> return own tokens for both regular user and admin
    if not username:
        return auth_info.username

    is_admin = await _is_system_admin(
        auth_info, mlrun.common.schemas.AuthorizationAction.read
    )

    # "*" wildcard -> system-admin only, returns all users
    if username == "*":
        if not is_admin:
            raise mlrun.errors.MLRunAccessDeniedError(
                "Only system admins can list tokens for all users"
            )
        return username

    # Specific username provided
    # Regular users can only query themselves
    if not is_admin and username != auth_info.username:
        raise mlrun.errors.MLRunAccessDeniedError(
            "Only system admins can read tokens for other users"
        )

    return username


async def _resolve_target_username_for_revoke_secret_tokens(
    auth_info: mlrun.common.schemas.AuthInfo,
    username: Optional[str],
) -> str:
    """
    Resolve the target username for REVOKE (delete) token operations.

    Regular users:
      - None, "", or self -> return auth_info.username (own token)
      - any other username -> raise MLRunAccessDeniedError

    Admin users:
      - None or "" -> return auth_info.username (own token)
      - specific username -> return that username
    """
    # No username provided (or username="") -> revoke own token for both regular user and admin
    if not username:
        return auth_info.username

    is_admin = await _is_system_admin(
        auth_info, mlrun.common.schemas.AuthorizationAction.delete
    )

    # Specific username provided
    # Regular users can only revoke their own tokens
    if not is_admin and username != auth_info.username:
        raise mlrun.errors.MLRunAccessDeniedError(
            "Only system admins can delete tokens for other users"
        )

    return username
