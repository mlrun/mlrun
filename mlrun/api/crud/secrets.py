# Copyright 2018 Iguazio
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
import enum
import json
import typing
import uuid

import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
import mlrun.utils.vault


class SecretsClientType(str, enum.Enum):
    schedules = "schedules"
    model_monitoring = "model-monitoring"
    service_accounts = "service-accounts"
    marketplace = "marketplace"


class Secrets(
    metaclass=mlrun.utils.singleton.Singleton,
):
    internal_secrets_key_prefix = "mlrun."
    # make it a subset of internal since key map are by definition internal
    key_map_secrets_key_prefix = f"{internal_secrets_key_prefix}map."

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
        if self._is_internal_project_secret_key(key) and not allow_internal_secrets:
            raise mlrun.errors.MLRunAccessDeniedError(
                f"Not allowed to create/update internal secrets (key starts with "
                f"{self.internal_secrets_key_prefix})"
            )

    def store_project_secrets(
        self,
        project: str,
        secrets: mlrun.api.schemas.SecretsData,
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

        if secrets.provider == mlrun.api.schemas.SecretProviderName.vault:
            # Init is idempotent and will do nothing if infra is already in place
            mlrun.utils.vault.init_project_vault_configuration(project)

            # If no secrets were passed, no need to touch the actual secrets.
            if secrets_to_store:
                mlrun.utils.vault.store_vault_project_secrets(project, secrets_to_store)
        elif secrets.provider == mlrun.api.schemas.SecretProviderName.kubernetes:
            if mlrun.api.utils.singletons.k8s.get_k8s():
                mlrun.api.utils.singletons.k8s.get_k8s().store_project_secrets(
                    project, secrets_to_store
                )
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
    ) -> mlrun.api.schemas.AuthSecretData:
        (
            username,
            access_key,
        ) = mlrun.api.utils.singletons.k8s.get_k8s().read_auth_secret(
            secret_name, raise_on_not_found=raise_on_not_found
        )
        return mlrun.api.schemas.AuthSecretData(
            provider=mlrun.api.schemas.SecretProviderName.kubernetes,
            username=username,
            access_key=access_key,
        )

    def store_auth_secret(
        self,
        secret: mlrun.api.schemas.AuthSecretData,
    ) -> str:
        if secret.provider != mlrun.api.schemas.SecretProviderName.kubernetes:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Storing auth secret is not implemented for provider {secret.provider}"
            )
        if not mlrun.api.utils.singletons.k8s.get_k8s():
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )
        return mlrun.api.utils.singletons.k8s.get_k8s().store_auth_secret(
            secret.username, secret.access_key
        )

    def delete_auth_secret(
        self,
        provider: mlrun.api.schemas.SecretProviderName,
        secret_name: str,
    ):
        if provider != mlrun.api.schemas.SecretProviderName.kubernetes:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Storing auth secret is not implemented for provider {provider}"
            )
        if not mlrun.api.utils.singletons.k8s.get_k8s():
            raise mlrun.errors.MLRunInternalServerError(
                "K8s provider cannot be initialized"
            )
        mlrun.api.utils.singletons.k8s.get_k8s().delete_auth_secret(secret_name)

    def delete_project_secrets(
        self,
        project: str,
        provider: mlrun.api.schemas.SecretProviderName,
        secrets: typing.Optional[typing.List[str]] = None,
        allow_internal_secrets: bool = False,
    ):
        if not allow_internal_secrets:
            if secrets:
                for secret_key in secrets:
                    if self._is_internal_project_secret_key(secret_key):
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

        if provider == mlrun.api.schemas.SecretProviderName.vault:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Delete secret is not implemented for provider {provider}"
            )
        elif provider == mlrun.api.schemas.SecretProviderName.kubernetes:
            if mlrun.api.utils.singletons.k8s.get_k8s():
                mlrun.api.utils.singletons.k8s.get_k8s().delete_project_secrets(
                    project, secrets
                )
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
        provider: mlrun.api.schemas.SecretProviderName,
        token: typing.Optional[str] = None,
        allow_internal_secrets: bool = False,
    ) -> mlrun.api.schemas.SecretKeysData:
        if provider == mlrun.api.schemas.SecretProviderName.vault:
            if not token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Vault list project secret keys request without providing token"
                )

            vault = mlrun.utils.vault.VaultStore(token)
            secret_values = vault.get_secrets(None, project=project)
            secret_keys = list(secret_values.keys())
        elif provider == mlrun.api.schemas.SecretProviderName.kubernetes:
            if token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Cannot specify token when requesting k8s secret keys"
                )

            if mlrun.api.utils.singletons.k8s.get_k8s():
                secret_keys = (
                    mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_keys(
                        project
                    )
                    or []
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
                    lambda key: not self._is_internal_project_secret_key(key),
                    secret_keys,
                )
            )

        return mlrun.api.schemas.SecretKeysData(
            provider=provider, secret_keys=secret_keys
        )

    def list_project_secrets(
        self,
        project: str,
        provider: mlrun.api.schemas.SecretProviderName,
        secrets: typing.Optional[typing.List[str]] = None,
        token: typing.Optional[str] = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
    ) -> mlrun.api.schemas.SecretsData:
        if provider == mlrun.api.schemas.SecretProviderName.vault:
            if not token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Vault list project secrets request without providing token"
                )

            vault = mlrun.utils.vault.VaultStore(token)
            secrets_data = vault.get_secrets(secrets, project=project)
        elif provider == mlrun.api.schemas.SecretProviderName.kubernetes:
            if not allow_secrets_from_k8s:
                raise mlrun.errors.MLRunAccessDeniedError(
                    "Not allowed to list secrets data from kubernetes provider"
                )
            secrets_data = (
                mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_data(
                    project, secrets
                )
            )

        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {provider}"
            )
        if not allow_internal_secrets:
            secrets_data = {
                key: value
                for key, value in secrets_data.items()
                if not self._is_internal_project_secret_key(key)
            }
        return mlrun.api.schemas.SecretsData(provider=provider, secrets=secrets_data)

    def delete_project_secret(
        self,
        project: str,
        provider: mlrun.api.schemas.SecretProviderName,
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
                    mlrun.api.schemas.SecretsData(
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
        provider: mlrun.api.schemas.SecretProviderName,
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

    def _resolve_project_secret_key(
        self,
        project: str,
        provider: mlrun.api.schemas.SecretProviderName,
        secret_key: str,
        token: typing.Optional[str] = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: typing.Optional[str] = None,
    ) -> typing.Tuple[bool, str]:
        if key_map_secret_key:
            if provider != mlrun.api.schemas.SecretProviderName.kubernetes:
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
        secrets: mlrun.api.schemas.SecretsData,
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
                if secrets.provider != mlrun.api.schemas.SecretProviderName.kubernetes:
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
            mlrun.api.schemas.SecretProviderName.kubernetes,
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

    def _is_internal_project_secret_key(self, key: str) -> bool:
        return key.startswith(self.internal_secrets_key_prefix)

    def _is_key_map_project_secret_key(self, key: str) -> bool:
        return key.startswith(self.key_map_secrets_key_prefix)

    @staticmethod
    def _generate_uuid() -> str:
        return str(uuid.uuid4())
