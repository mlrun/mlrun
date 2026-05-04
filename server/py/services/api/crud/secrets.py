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

import asyncio
import enum
import json
import typing
import uuid
from collections import defaultdict

from fastapi.concurrency import run_in_threadpool

import mlrun.auth.utils
import mlrun.common
import mlrun.common.constants
import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.errors
import mlrun.k8s_utils
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
import mlrun.utils.vault
from mlrun.config import config as mlconf
from mlrun.utils import logger

import framework.utils.clients.iguazio.v4
import framework.utils.singletons.k8s
import services.api
import services.api.utils.events.events_factory as events_factory


class SecretsClientType(enum.StrEnum):
    schedules = "schedules"
    model_monitoring = "model-monitoring"
    service_accounts = "service-accounts"
    hub = "hub"
    notifications = "notifications"
    datastore_profiles = "datastore-profiles"


class Secrets(
    metaclass=mlrun.utils.singleton.Singleton,
):
    internal_secrets_key_prefix = "mlrun."
    # make it a subset of internal since key map are by definition internal
    key_map_secrets_key_prefix = f"{internal_secrets_key_prefix}map."

    def __init__(self):
        if mlconf.secret_stores.test_mode_mock_secrets:
            logger.warning("***** USING SECRETS IN TEST MODE *****")
            logger.warning(
                "***** Secrets are kept in-memory. Only use this mode for testing *****"
            )
            self.secrets_provider = mlrun.common.secrets.InMemorySecretProvider()
        else:
            self.secrets_provider = framework.utils.singletons.k8s.get_k8s_helper()

    @property
    def secrets_provider(self) -> mlrun.common.secrets.SecretProviderInterface:
        return self._secrets_provider

    @secrets_provider.setter
    def secrets_provider(self, provider: mlrun.common.secrets.SecretProviderInterface):
        self._secrets_provider = provider

    def generate_client_project_secret_key(
        self, client_type: SecretsClientType, name: str, subtype=None
    ):
        key_name = f"{self.internal_secrets_key_prefix}{client_type.value}.{name}"
        if subtype:
            key_name = f"{key_name}.{subtype}"
        return key_name

    def generate_client_key_map_project_secret_key(
        self, client_type: SecretsClientType
    ):
        return f"{self.key_map_secrets_key_prefix}{client_type.value}"

    @staticmethod
    def validate_project_secret_key_regex(
        key: str, raise_on_failure: bool = True
    ) -> bool:
        return mlrun.utils.helpers.verify_field_regex(
            "secret.key", key, mlrun.utils.regex.secret_key, raise_on_failure
        )

    def validate_internal_project_secret_key_allowed(
        self, key: str, allow_internal_secrets: bool = False
    ):
        if self.is_internal_project_secret_key(key) and not allow_internal_secrets:
            raise mlrun.errors.MLRunAccessDeniedError(
                f"Not allowed to create/update internal secrets (key starts with "
                f"{self.internal_secrets_key_prefix})"
            )

    def store_project_secrets(
        self,
        project: str,
        secrets: mlrun.common.schemas.SecretsData,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
        allow_storing_key_maps: bool = False,
    ):
        """
        When secret keys are coming from other object identifiers, which may not be valid secret keys, use
        key_map_secret_key.
        Note that when it's used you'll need to get and delete secrets using the get_project_secret and
        delete_project_secret list_project_secrets won't do any operation on the data and delete_project_secrets won't
        handle cleaning the key map
        """
        secrets_to_store = self._validate_and_enrich_project_secrets_to_store(
            project,
            secrets,
            allow_internal_secrets,
            key_map_secret_key,
            allow_storing_key_maps,
        )

        if secrets.provider == mlrun.common.schemas.SecretProviderName.vault:
            # Init is idempotent and will do nothing if infra is already in place
            mlrun.utils.vault.init_project_vault_configuration(project)

            # If no secrets were passed, no need to touch the actual secrets.
            if secrets_to_store:
                mlrun.utils.vault.store_vault_project_secrets(project, secrets_to_store)
        elif secrets.provider == mlrun.common.schemas.SecretProviderName.kubernetes:
            if self.secrets_provider:
                (
                    secret_name,
                    action,
                ) = self.secrets_provider.store_project_secrets(
                    project, secrets_to_store
                )
                secret_keys = [secret_name for secret_name in secrets_to_store.keys()]

                if action:
                    events_client = events_factory.EventsFactory().get_events_client()
                    event = events_client.generate_project_secret_event(
                        project=project,
                        secret_name=secret_name,
                        secret_keys=secret_keys,
                        action=action,
                    )
                    events_client.emit(event)

            else:
                raise mlrun.errors.MLRunInternalServerError(
                    "K8s provider cannot be initialized"
                )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {secrets.provider}"
            )

    def read_auth_secret(
        self, secret_name, raise_on_not_found=False
    ) -> mlrun.common.schemas.AuthSecretData:
        (
            username,
            access_key,
        ) = self.secrets_provider.read_auth_secret(
            secret_name, raise_on_not_found=raise_on_not_found
        )
        return mlrun.common.schemas.AuthSecretData(
            provider=mlrun.common.schemas.SecretProviderName.kubernetes,
            username=username,
            access_key=access_key,
        )

    def store_auth_secret(
        self,
        secret: mlrun.common.schemas.AuthSecretData,
    ) -> str:
        if secret.provider != mlrun.common.schemas.SecretProviderName.kubernetes:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Storing auth secret is not implemented for provider {secret.provider}"
            )
        if not self.secrets_provider:
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )

        # ignore the returned action as we don't need to emit an event for auth secrets (they are internal)
        (
            auth_secret_name,
            _,
        ) = self.secrets_provider.store_auth_secret(secret.username, secret.access_key)

        return auth_secret_name

    def delete_auth_secret(
        self,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_name: str,
    ):
        if provider != mlrun.common.schemas.SecretProviderName.kubernetes:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Storing auth secret is not implemented for provider {provider}"
            )
        if not self.secrets_provider:
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )
        self.secrets_provider.delete_auth_secret(secret_name)

    def delete_project_secrets(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secrets: list[str] | None = None,
        allow_internal_secrets: bool = False,
    ):
        if not allow_internal_secrets:
            if secrets:
                for secret_key in secrets:
                    if self.is_internal_project_secret_key(secret_key):
                        raise mlrun.errors.MLRunAccessDeniedError(
                            f"Not allowed to delete internal secrets (key starts with "
                            f"{self.internal_secrets_key_prefix})"
                        )
            else:
                # When secrets are not provided the default behavior will be to delete them all, but if internal secrets
                # are not allowed, we don't want to delete them, so we list the non internal keys
                secrets = self.list_project_secret_keys(
                    project, provider, allow_internal_secrets=False
                ).secret_keys
                if not secrets:
                    # nothing to remove - return
                    return

        if provider == mlrun.common.schemas.SecretProviderName.vault:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Delete secret is not implemented for provider {provider}"
            )
        elif provider == mlrun.common.schemas.SecretProviderName.kubernetes:
            if self.secrets_provider:
                (
                    secret_name,
                    action,
                ) = self.secrets_provider.delete_project_secrets(project, secrets)

                if action:
                    events_client = events_factory.EventsFactory().get_events_client()
                    event = events_client.generate_project_secret_event(
                        project=project,
                        secret_name=secret_name,
                        secret_keys=secrets,
                        action=action,
                    )
                    events_client.emit(event)

            else:
                raise mlrun.errors.MLRunInternalServerError(
                    "K8s provider cannot be initialized"
                )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {provider}"
            )

    def list_project_secret_keys(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        token: str | None = None,
        allow_internal_secrets: bool = False,
    ) -> mlrun.common.schemas.SecretKeysData:
        if provider == mlrun.common.schemas.SecretProviderName.vault:
            if not token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Vault list project secret keys request without providing token"
                )

            vault = mlrun.utils.vault.VaultStore(token)
            secret_values = vault.get_secrets(None, project=project)
            secret_keys = list(secret_values.keys())
        elif provider == mlrun.common.schemas.SecretProviderName.kubernetes:
            if token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Cannot specify token when requesting k8s secret keys"
                )

            if self.secrets_provider:
                secret_keys = (
                    self.secrets_provider.get_project_secret_keys(project) or []
                )
            else:
                raise mlrun.errors.MLRunInternalServerError(
                    "K8s provider cannot be initialized"
                )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {provider}"
            )
        if not allow_internal_secrets:
            secret_keys = list(
                filter(
                    lambda key: not self.is_internal_project_secret_key(key),
                    secret_keys,
                )
            )

        return mlrun.common.schemas.SecretKeysData(
            provider=provider, secret_keys=secret_keys
        )

    def list_project_secrets(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secrets: list[str] | None = None,
        token: str | None = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
    ) -> mlrun.common.schemas.SecretsData:
        if provider == mlrun.common.schemas.SecretProviderName.vault:
            if not token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Vault list project secrets request without providing token"
                )

            vault = mlrun.utils.vault.VaultStore(token)
            secrets_data = vault.get_secrets(secrets, project=project)
        elif provider == mlrun.common.schemas.SecretProviderName.kubernetes:
            if not allow_secrets_from_k8s:
                raise mlrun.errors.MLRunAccessDeniedError(
                    "Not allowed to list secrets data from kubernetes provider"
                )
            secrets_data = self.secrets_provider.get_project_secret_data(
                project, secrets
            )

        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {provider}"
            )
        if not allow_internal_secrets:
            secrets_data = {
                key: value
                for key, value in secrets_data.items()
                if not self.is_internal_project_secret_key(key)
            }
        return mlrun.common.schemas.SecretsData(provider=provider, secrets=secrets_data)

    def delete_project_secret(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_key: str,
        token: str | None = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
    ):
        from_key_map, secret_key_to_remove = self._resolve_project_secret_key(
            project,
            provider,
            secret_key,
            token,
            allow_secrets_from_k8s,
            allow_internal_secrets,
            key_map_secret_key,
        )
        self.delete_project_secrets(
            project, provider, [secret_key_to_remove], allow_internal_secrets
        )
        if from_key_map:
            # clean key from key map
            key_map = self._get_project_secret_key_map(project, key_map_secret_key)
            del key_map[secret_key]
            if key_map:
                self.store_project_secrets(
                    project,
                    mlrun.common.schemas.SecretsData(
                        provider=provider,
                        secrets={key_map_secret_key: json.dumps(key_map)},
                    ),
                    allow_internal_secrets=True,
                    allow_storing_key_maps=True,
                )
            else:
                self.delete_project_secrets(
                    project, provider, [key_map_secret_key], allow_internal_secrets=True
                )

    def get_project_secret(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_key: str,
        token: str | None = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
    ) -> str | None:
        from_key_map, secret_key = self._resolve_project_secret_key(
            project,
            provider,
            secret_key,
            token,
            allow_secrets_from_k8s,
            allow_internal_secrets,
            key_map_secret_key,
        )
        secrets_data = self.list_project_secrets(
            project,
            provider,
            [secret_key],
            token,
            allow_secrets_from_k8s,
            allow_internal_secrets,
        )
        return secrets_data.secrets.get(secret_key)

    def is_internal_project_secret_key(self, key: str) -> bool:
        return key.startswith(self.internal_secrets_key_prefix)

    def store_secret_tokens(
        self,
        secret_tokens: list[mlrun.common.schemas.SecretToken],
        auth_info: mlrun.common.schemas.AuthInfo,
        force: bool = False,
    ) -> mlrun.common.schemas.StoreSecretTokensResponse:
        """
        Validate and store offline tokens as Kubernetes secrets.

        :param secret_tokens: List of SecretToken objects to store.
        :param force: Whether to force update existing tokens.
        :param auth_info: Authentication information of the user storing the tokens.
        :return: StoreSecretTokensResponse object with created, updated, and skipped tokens.
        """
        if not secret_tokens:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Failed to store secret tokens – no tokens provided"
            )

        logger.debug(
            "Storing secret tokens",
            username=auth_info.username,
            token_count=len(secret_tokens),
        )

        # Extract and validate tokens info
        tokens_values = mlrun.auth.utils.extract_and_validate_tokens_info(
            secret_tokens=secret_tokens, authenticated_id=auth_info.user_id
        )

        # TODO: move init iguazio_client (ML-11077)
        iguazio_client = framework.utils.clients.iguazio.v4.Client()

        # We validate the offline tokens by sending it to Iguazio for verification.
        iguazio_client.refresh_access_tokens(secret_tokens)

        token_actions = defaultdict(list)

        for token_name, token_info in tokens_values.items():
            token = token_info["token"]
            expiration = token_info["token_exp"]
            issued_at = token_info["token_iat"]

            action = self.secrets_provider.store_user_token_secret(
                auth_info=auth_info,
                token_name=token_name,
                token=token,
                expiration=expiration,
                issued_at=issued_at,
                force=force,
            )
            if action is not None:
                token_actions[action].append(token_name)

        if token_actions:
            logger.debug(
                "Finished storing tokens",
                created_tokens=token_actions[
                    mlrun.common.schemas.SecretEventActions.created
                ],
                updated_tokens=token_actions[
                    mlrun.common.schemas.SecretEventActions.updated
                ],
            )

        return mlrun.common.schemas.StoreSecretTokensResponse(
            created_tokens=token_actions[
                mlrun.common.schemas.SecretEventActions.created
            ],
            updated_tokens=token_actions[
                mlrun.common.schemas.SecretEventActions.updated
            ],
        )

    def list_secret_tokens(
        self,
        auth_info: mlrun.common.schemas.AuthInfo,
        username: str | None = None,
    ) -> mlrun.common.schemas.ListSecretTokensResponse:
        """
        List offline token secrets stored in Kubernetes.

        By default, this lists tokens for the authenticated user.
        Admins can list tokens for other users by providing a username.

        :param auth_info: Authentication information of the requesting user.
        :param username: Target username to list tokens for. If None or matches
                         auth_info.username, lists the authenticated user's tokens.
                         Use "*" to list all users' tokens (admin only).
        :return: ListSecretTokensResponse containing token names and expirations.
        """
        target_user_id = self._get_user_id(auth_info, username)

        secret_tokens = self.secrets_provider.list_user_token_secrets(
            user_id=target_user_id,
        )

        return mlrun.common.schemas.ListSecretTokensResponse(
            secret_tokens=secret_tokens
        )

    def _delete_single_token(
        self,
        target_user_id: str,
        target_username: str,
        token_name: str,
        iguazio_client: "framework.utils.clients.iguazio.v4.Client",
        request_headers: dict[str, str] | None,
        skip_revocation: bool = False,
    ) -> None:
        """
        Delete a single token: get value, optionally revoke in Iguazio, delete from K8s.

        :param target_user_id: The user_id of the token owner.
        :param target_username: The username of the token owner (for logging).
        :param token_name: The name of the token to delete.
        :param iguazio_client: The Iguazio client to use for revocation.
        :param request_headers: Request headers for authenticating with Iguazio.
        :param skip_revocation: If True, skip revoking the token via Iguazio and only delete
                                the K8s secret. Used in bulk delete during user deletion flow
                                since tokens are invalidated when the user is deleted anyway.
        :raises mlrun.errors.MLRunNotFoundError: If the token is not found.
        :raises mlrun.errors.MLRunRuntimeError: If K8s deletion fails after revocation.
        """
        if not skip_revocation:
            # Get the offline token string
            token = self.secrets_provider.get_user_token_secret_value(
                user_id=target_user_id,
                token_name=token_name,
            )

            # Revoke via Iguazio
            iguazio_client.revoke_offline_token(token, request_headers)

        # Delete the Kubernetes secret
        try:
            self.secrets_provider.delete_user_token_secret(
                user_id=target_user_id,
                token_name=token_name,
            )
        except Exception as exc:
            logger.error(
                "Failed to delete token secret",
                target_user_id=target_user_id,
                target_username=target_username,
                token_name=token_name,
                exc=mlrun.errors.err_to_str(exc),
            )
            err_msg = (
                f"Failed to delete K8s secret for token '{token_name}'"
                if skip_revocation
                else f"Token '{token_name}' revoked but failed to delete associated K8s secret"
            )
            raise mlrun.errors.MLRunRuntimeError(err_msg) from exc

    def delete_secret_token(
        self,
        token_name: str,
        username: str,
        auth_info: mlrun.common.schemas.AuthInfo,
    ) -> mlrun.common.schemas.DeleteSecretTokenResponse:
        """
        Delete a stored offline token for a user and its corresponding Kubernetes secret.

        This method performs two actions:
        1. Calls the Iguazio management service to revoke the offline token.
        2. Removes the Kubernetes secret associated with the token.

        :param token_name:
            Logical name of the token to delete (used in the Kubernetes secret name).
        :param username:
            The username of the user who owns the token to be deleted.
            For regular users, this must be their own username.
            For system admins, this can be any user's username.
        :param auth_info:
            Authentication information of the requesting user.
        :return: DeleteSecretTokenResponse with deleted=True if token was deleted,
                 or deleted=False if token was not found.
        """

        target_user_id = self._get_user_id(auth_info, username)

        logger.debug(
            "Revoking secret token for user",
            target_user_id=target_user_id,
            target_username=username,
            requesting_user=auth_info.username,
        )

        # TODO: move init iguazio_client (ML-11077)
        iguazio_client = framework.utils.clients.iguazio.v4.Client()

        try:
            self._delete_single_token(
                target_user_id=target_user_id,
                target_username=username,
                token_name=token_name,
                iguazio_client=iguazio_client,
                request_headers=auth_info.request_headers,
            )
        except mlrun.errors.MLRunNotFoundError:
            logger.warning(
                "Token not found, nothing to revoke",
                target_user_id=target_user_id,
                target_username=username,
                token_name=token_name,
            )
            return mlrun.common.schemas.DeleteSecretTokenResponse(
                deleted=False, username=username
            )

        logger.debug(
            "Finished revoking secret token for user",
            target_user_id=target_user_id,
            target_username=username,
            token_name=token_name,
        )
        return mlrun.common.schemas.DeleteSecretTokenResponse(
            deleted=True, username=username
        )

    async def delete_secret_tokens(
        self,
        username: str,
        auth_info: mlrun.common.schemas.AuthInfo,
    ) -> mlrun.common.schemas.DeleteSecretTokensResponse:
        """
        Delete all Kubernetes secrets storing tokens for a user.

        Deletes each token's K8s secret in parallel (bounded by
        secret_stores.kubernetes.concurrent_token_deletions).
        Failures are collected and returned without stopping other deletions.

        Token revocation is intentionally skipped — this endpoint is designed for the
        user-deletion flow where the user is already deactivated and Keycloak removal
        invalidates all tokens. If this endpoint is ever reused outside that flow,
        skip_revocation should become a caller-controlled flag.

        :param username:
            The username of the user whose tokens should be deleted.
            For regular users, this must be their own username.
            For system admins, this can be any user's username.
        :param auth_info:
            Authentication information of the requesting user.
        :return: DeleteSecretTokensResponse with deleted_count and any failed_tokens.
        """
        target_user_id = await run_in_threadpool(self._get_user_id, auth_info, username)

        logger.debug(
            "Deleting all secret tokens for user",
            target_user_id=target_user_id,
            target_username=username,
            requesting_user=auth_info.username,
        )

        tokens: list[mlrun.common.schemas.SecretTokenInfo] = await run_in_threadpool(
            self.secrets_provider.list_user_token_secrets,
            user_id=target_user_id,
        )

        if not tokens:
            return mlrun.common.schemas.DeleteSecretTokensResponse(
                deleted_count=0, failed_tokens=[], username=username
            )

        # TODO: move init iguazio_client (ML-11077)
        iguazio_client = framework.utils.clients.iguazio.v4.Client()

        # TODO: Replace per-token deletion with delete_collection_namespaced_secret
        # This would reduce N K8s API calls to a single collection delete. (IG4-1510)
        semaphore = asyncio.Semaphore(
            mlrun.mlconf.secret_stores.kubernetes.concurrent_token_deletions
        )

        async def _delete_with_semaphore(token_name: str):
            async with semaphore:
                await run_in_threadpool(
                    self._delete_single_token,
                    target_user_id=target_user_id,
                    target_username=username,
                    token_name=token_name,
                    iguazio_client=iguazio_client,
                    request_headers=auth_info.request_headers,
                    # User is already deactivated and Keycloak removal invalidates tokens
                    skip_revocation=True,
                )

        results = await asyncio.gather(
            *[_delete_with_semaphore(token_info.name) for token_info in tokens],
            return_exceptions=True,
        )

        deleted_count = 0
        failed_tokens: list[str] = []

        for i, result in enumerate(results):
            token_name = tokens[i].name
            if isinstance(result, Exception):
                failed_tokens.append(token_name)
            else:
                deleted_count += 1

        if failed_tokens:
            logger.warning(
                "Some tokens failed to delete",
                target_user_id=target_user_id,
                target_username=username,
                deleted_count=deleted_count,
                failed_count=len(failed_tokens),
            )

        logger.debug(
            "Finished deleting secret tokens for user",
            target_user_id=target_user_id,
            target_username=username,
            deleted_count=deleted_count,
            failed_count=len(failed_tokens),
        )

        return mlrun.common.schemas.DeleteSecretTokensResponse(
            deleted_count=deleted_count, failed_tokens=failed_tokens, username=username
        )

    def get_secret_token(
        self,
        token_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
    ) -> mlrun.common.schemas.SecretToken:
        """
        Get a specific offline token stored for a user by token name.

        :param token_name: Name of the token to retrieve.
        :param auth_info: Authentication information of the user.
        :return: SecretToken object containing the token name and token value.
        :raises mlrun.errors.MLRunNotFoundError: If the token does not exist for the user.
        :raises mlrun.errors.MLRunRuntimeError: If reading or decoding the token fails.
        """

        token_value = self.secrets_provider.get_user_token_secret_value(
            user_id=auth_info.user_id,
            token_name=token_name,
        )

        return mlrun.common.schemas.SecretToken(
            name=token_name,
            token=token_value,
        )

    def _get_user_id(
        self,
        auth_info: mlrun.common.schemas.AuthInfo,
        username: str | None,
    ) -> str:
        """
        Get the user_id for token operations.

        If the username is None, empty, or matches the authenticated user's username,
        returns the authenticated user's user_id directly.

        If the username is "*", returns "*" to indicate all users (for list operations).

        Otherwise, fetches the user_id from the Iguazio API (blocking I/O).

        :param auth_info: Authentication information of the requesting user.
        :param username: Target username. Can be None, "", "*", or a specific username.
        :return: The user_id, or "*" for all users.
        :raises mlrun.errors.MLRunNotFoundError: If the username cannot be found.
        """
        # No username provided or matches self -> use authenticated user's user_id
        if not username or username == auth_info.username:
            return auth_info.user_id

        # Wildcard for all users (list operation)
        if username == "*":
            return "*"

        # Different user - fetch user_id from Iguazio API
        iguazio_client = framework.utils.clients.iguazio.v4.Client()
        return iguazio_client.get_user_id_by_username(username, auth_info)

    def _resolve_project_secret_key(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_key: str,
        token: str | None = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
    ) -> tuple[bool, str]:
        if key_map_secret_key:
            if provider != mlrun.common.schemas.SecretProviderName.kubernetes:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Secret using key map is not implemented for provider {provider}"
                )
            if self._is_project_secret_stored_in_key_map(secret_key):
                secrets_data = self.list_project_secrets(
                    project,
                    provider,
                    [key_map_secret_key],
                    token,
                    allow_secrets_from_k8s,
                    allow_internal_secrets,
                )
                if secrets_data.secrets.get(key_map_secret_key):
                    key_map = json.loads(secrets_data.secrets[key_map_secret_key])
                    if secret_key in key_map:
                        return True, key_map[secret_key]
        return False, secret_key

    def _validate_and_enrich_project_secrets_to_store(
        self,
        project: str,
        secrets: mlrun.common.schemas.SecretsData,
        allow_internal_secrets: bool = False,
        key_map_secret_key: str | None = None,
        allow_storing_key_maps: bool = False,
    ):
        secrets_to_store = secrets.secrets.copy()
        if secrets_to_store:
            for secret_key in secrets_to_store.keys():
                # key map is there to allow using invalid secret keys
                if not key_map_secret_key:
                    self.validate_project_secret_key_regex(secret_key)
                self.validate_internal_project_secret_key_allowed(
                    secret_key, allow_internal_secrets
                )
                if (
                    self._is_key_map_project_secret_key(secret_key)
                    and not allow_storing_key_maps
                ):
                    raise mlrun.errors.MLRunAccessDeniedError(
                        f"Not allowed to create/update key map (key starts with "
                        f"{self.key_map_secrets_key_prefix})"
                    )
            if key_map_secret_key:
                if (
                    secrets.provider
                    != mlrun.common.schemas.SecretProviderName.kubernetes
                ):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Storing secret using key map is not implemented for provider {secrets.provider}"
                    )
                if not self._is_key_map_project_secret_key(key_map_secret_key):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Key map secret key must start with: {self.key_map_secrets_key_prefix}"
                    )
                if not allow_internal_secrets:
                    raise mlrun.errors.MLRunAccessDeniedError(
                        f"Not allowed to create/update internal secrets (key starts with "
                        f"{self.internal_secrets_key_prefix})"
                    )
                self.validate_project_secret_key_regex(key_map_secret_key)
                secrets_to_store_in_key_map = [
                    secret_key
                    for secret_key in secrets_to_store.keys()
                    if self._is_project_secret_stored_in_key_map(secret_key)
                ]
                if secrets_to_store_in_key_map:
                    key_map = (
                        self._get_project_secret_key_map(project, key_map_secret_key)
                        or {}
                    )
                    key_map.update(
                        {
                            secret_key: self._generate_uuid()
                            for secret_key in secrets_to_store_in_key_map
                            if secret_key not in key_map
                        }
                    )
                    updated_secrets_to_store = {}
                    for key, value in secrets_to_store.items():
                        if key in secrets_to_store_in_key_map:
                            updated_secrets_to_store[key_map[key]] = value
                        else:
                            updated_secrets_to_store[key] = value
                    updated_secrets_to_store[key_map_secret_key] = json.dumps(key_map)
                    secrets_to_store = updated_secrets_to_store
        return secrets_to_store

    def _get_project_secret_key_map(
        self,
        project: str,
        key_map_secret_key: str,
    ) -> dict | None:
        secrets_data = self.list_project_secrets(
            project,
            mlrun.common.schemas.SecretProviderName.kubernetes,
            [key_map_secret_key],
            allow_secrets_from_k8s=True,
            allow_internal_secrets=True,
        )
        value = secrets_data.secrets.get(key_map_secret_key)
        if value:
            value = json.loads(value)
        return value

    def _is_project_secret_stored_in_key_map(self, key: str) -> bool:
        # Key map are only used for invalid keys
        return not self.validate_project_secret_key_regex(key, raise_on_failure=False)

    def _is_key_map_project_secret_key(self, key: str) -> bool:
        return key.startswith(self.key_map_secrets_key_prefix)

    @staticmethod
    def _generate_uuid() -> str:
        return str(uuid.uuid4())


def get_project_secret_provider(project: str) -> typing.Callable:
    """Implement secret provider for handle the related project secret on the API side.

    :param project: Project name.

    :return: A secret provider function.
    """

    def secret_provider(key: str):
        return services.api.crud.secrets.Secrets().get_project_secret(
            project=project,
            provider=mlrun.common.schemas.secret.SecretProviderName.kubernetes,
            allow_secrets_from_k8s=True,
            secret_key=key,
        )

    return secret_provider
