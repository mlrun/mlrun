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

import enum
import json
import typing
import uuid
from collections import defaultdict

import jwt

import mlrun.common
import mlrun.common.constants
import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.errors
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


class SecretsClientType(str, enum.Enum):
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
        key_map_secret_key: typing.Optional[str] = None,
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
        secrets: typing.Optional[list[str]] = None,
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
        token: typing.Optional[str] = None,
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
        secrets: typing.Optional[list[str]] = None,
        token: typing.Optional[str] = None,
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
        token: typing.Optional[str] = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: typing.Optional[str] = None,
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
        token: typing.Optional[str] = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: typing.Optional[str] = None,
    ) -> typing.Optional[str]:
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
        tokens_values = self._extract_and_validate_tokens_info(
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

            action = self.secrets_provider.store_user_token_secret(
                username=auth_info.username,
                token_name=token_name,
                token=token,
                expiration=expiration,
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
        authenticated_username: str,
    ) -> mlrun.common.schemas.ListSecretTokensResponse:
        """
        List all offline tokens stored for the authenticated user.

        :param authenticated_username: Username whose tokens will be listed.
        :return: ListSecretTokensResponse containing token names and expirations.
        """
        logger.debug(
            "Listing secret tokens for user",
            username=authenticated_username,
        )

        secret_tokens = self.secrets_provider.list_user_token_secrets(
            username=authenticated_username,
        )

        logger.debug(
            "Finished listing secret tokens",
            username=authenticated_username,
            token_count=len(secret_tokens),
        )

        return mlrun.common.schemas.ListSecretTokensResponse(
            secret_tokens=secret_tokens
        )

    def revoke_secret_token(
        self,
        token_name: str,
        authenticated_username: str,
        request_headers: typing.Optional[dict[str, str]] = None,
    ):
        """
        Revoke a stored offline token for a user and delete its corresponding Kubernetes secret.

        This method performs two actions:
        1. Calls the Iguazio management service to revoke the offline token.
        2. Removes the Kubernetes secret named `mlrun-auth-<username>-<token_name>`
           associated with the token.

        :param token_name:
            Logical name of the token to revoke (used in the Kubernetes secret name).
        :param authenticated_username:
            The username of the authenticated user who owns the token.
        :param request_headers:
            Optional request headers (e.g., containing the user's access token)
            to authenticate with the Iguazio management service.
        """
        logger.debug(
            "Revoking secret token for user",
            username=authenticated_username,
            token_name=token_name,
        )

        try:
            # Get the offline token string
            token = self.secrets_provider.get_user_token_secret_value(
                username=authenticated_username,
                token_name=token_name,
            )
        except mlrun.errors.MLRunNotFoundError:
            logger.warning(
                "Token not found, nothing to revoke",
                username=authenticated_username,
                token_name=token_name,
            )
            return

        # Revoke via Iguazio
        # TODO: move init iguazio_client (ML-11077)
        iguazio_client = framework.utils.clients.iguazio.v4.Client()
        iguazio_client.revoke_offline_token(token, request_headers)

        # Delete the Kubernetes secret
        try:
            self.secrets_provider.delete_user_token_secret(
                username=authenticated_username,
                token_name=token_name,
            )
        except Exception as exc:
            logger.error(
                "Token revoked but failed to delete associated secret",
                username=authenticated_username,
                token_name=token_name,
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunRuntimeError(
                f"Token '{token_name}' revoked, but failed to delete associated secret"
            ) from exc

        logger.debug(
            "Finished revoking secret token for user",
            username=authenticated_username,
            token_name=token_name,
        )

    def get_secret_token(
        self,
        token_name: str,
        authenticated_username: str,
    ) -> mlrun.common.schemas.SecretToken:
        """
        Get a specific offline token stored for the authenticated user by token name.

        :param token_name: Name of the token to retrieve.
        :param authenticated_username: Username whose token will be retrieved.
        :return: SecretToken object containing the token name and token value.
        :raises mlrun.errors.MLRunNotFoundError: If the token does not exist for the user.
        :raises mlrun.errors.MLRunRuntimeError: If reading or decoding the token fails.
        """

        token_value = self.secrets_provider.get_user_token_secret_value(
            username=authenticated_username,
            token_name=token_name,
        )

        return mlrun.common.schemas.SecretToken(
            name=token_name,
            token=token_value,
        )

    def _extract_and_validate_tokens_info(
        self,
        secret_tokens: list[mlrun.common.schemas.SecretToken],
        authenticated_id: str,
    ):
        token_values = {}
        for secret_token in secret_tokens:
            token_name = secret_token.name

            # Validate name is provided and not duplicate
            if secret_token.name and secret_token.name not in token_values:
                decoded_token = self._decode_offline_token(
                    secret_token.name, secret_token.token
                )

                # Validate token expiration existence
                if not decoded_token.get("exp"):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Offline token '{token_name}' is missing the 'exp' (expiration) claim"
                    )
                # Validate token subject existence
                if not decoded_token.get("sub"):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Offline token '{token_name}' is missing the 'sub' (subject) claim"
                    )

                # Validate token belongs to the authenticated user
                token_sub = decoded_token.get("sub")
                if token_sub != authenticated_id:
                    mlrun.utils.logger.warning(
                        "Offline token subject does not match the authenticated user",
                        token_name=token_name,
                        token_sub=token_sub,
                        user_id=authenticated_id,
                    )
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Offline token '{token_name}' does not match the authenticated user ID. "
                        "Stored tokens can only belong to the authenticated user."
                    )

                # Store token info
                token_values[secret_token.name] = {
                    "token_exp": decoded_token.get("exp"),
                    "token": secret_token.token,
                }
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Invalid or duplicate token name '{secret_token.name}' found in request payload"
                )
        return token_values

    @staticmethod
    def _decode_offline_token(token_name: str, token: str) -> dict:
        try:
            # The token is expected to be a JWT. We don't verify its signature here, because it has already been
            # verified earlier during the refresh_access_token call.
            return jwt.decode(token, options={"verify_signature": False})
        except jwt.DecodeError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Failed to decode offline token '{token_name}'"
            ) from exc
        except Exception as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unexpected error decoding token '{token_name}'"
            ) from exc

    def _resolve_project_secret_key(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_key: str,
        token: typing.Optional[str] = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: typing.Optional[str] = None,
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
        key_map_secret_key: typing.Optional[str] = None,
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
    ) -> typing.Optional[dict]:
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

    @staticmethod
    def mount_secret_token_to_runtime(
        runtime: mlrun.runtimes.base.BaseRuntime, token_name: str, username: str
    ):
        # Validation that the secret exists is done in the ServerSideLauncher
        secret = framework.utils.singletons.k8s.get_k8s_helper()._get_user_token_secret(
            username=username, token_name=token_name
        )

        # In case the secret was not found (which should not happen because of the prior validation), we do not mount it
        if secret:
            runtime.apply(
                mlrun.mounts.mount_secret(
                    secret.metadata.name,
                    mount_path=mlrun.common.constants.MLRUN_JOB_AUTH_SECRET_PATH,
                    items=[
                        {
                            "key": "tokensFile",
                            "path": mlrun.common.constants.MLRUN_JOB_AUTH_SECRET_FILE,
                        }
                    ],
                )
            )
        return runtime


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
