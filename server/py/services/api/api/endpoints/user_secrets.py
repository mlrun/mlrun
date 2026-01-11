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


async def _resolve_target_username(
    auth_info: mlrun.common.schemas.AuthInfo,
    username: Optional[str],
    action: mlrun.common.schemas.AuthorizationAction,
) -> Optional[str]:
    """
    Resolve the target username for token operations.

    - If `username` is provided, only system admins are allowed to use it.
    - If `username` is not provided:
      - System admin with 'read' action: returns None (all users)
      - Otherwise: returns the authenticated user's username

    :param auth_info: Authentication information of the user.
    :param username: Optional username parameter from the request.
    :param action: The authorization action (read, delete, etc.).
    :return: The resolved target username, or None for all users.
    :raises MLRunAccessDeniedError: If a non-admin tries to use the username parameter.
    """
    if username is not None:
        # Only system admins can specify a username
        is_admin = await _is_system_admin(auth_info, action)
        if not is_admin:
            raise mlrun.errors.MLRunAccessDeniedError(
                f"Only system admins can {action.value} tokens for other users"
            )
        return username

    # No username provided
    is_admin = await _is_system_admin(auth_info, action)
    if is_admin and action == mlrun.common.schemas.AuthorizationAction.read:
        # Admin listing: return None to get all users' tokens
        return None

    # Regular user or admin revoking: use their own username
    return auth_info.username


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
        description="Optional username to filter tokens. Only system admins can use this parameter.",
    ),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        framework.api.deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
):
    """
    List secret tokens.

    Authorization logic:
    - If `username` is provided: only system admins can use this parameter
      - Lists tokens for the specified user
    - If `username` is not provided:
      - System admin: lists tokens for ALL users
      - Regular user: lists only their own tokens
    """
    if username == "":
        raise mlrun.errors.MLRunBadRequestError("Username cannot be an empty string.")
    target_username = await _resolve_target_username(
        auth_info, username, mlrun.common.schemas.AuthorizationAction.read
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
    db_session: Session = fastapi.Depends(framework.api.deps.get_db_session),
):
    """
    Revoke a secret token.

    Authorization logic:
    - If `username` is provided: only system admins can use this parameter
      - Revokes the specified user's token
    - If `username` is not provided: revokes the authenticated user's token

    Returns:
        RevokeSecretTokenResponse with revoked=True if token was revoked,
        or revoked=False if token was not found.
    """
    if username == "":
        raise mlrun.errors.MLRunBadRequestError("Username cannot be an empty string.")
    target_username = await _resolve_target_username(
        auth_info, username, mlrun.common.schemas.AuthorizationAction.delete
    )
    return await run_in_threadpool(
        services.api.crud.Secrets().revoke_secret_token,
        name,
        target_username,
        auth_info.request_headers,
    )
