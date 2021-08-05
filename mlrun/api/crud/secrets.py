import typing

import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.regex
import mlrun.utils.singleton
import mlrun.utils.vault


class Secrets(metaclass=mlrun.utils.singleton.Singleton,):
    internal_secrets_key_prefix = "mlrun."

    def generate_schedule_secret_key(self, schedule_name: str):
        return f"{self.internal_secrets_key_prefix}schedules.{schedule_name}"

    def store_secrets(
        self,
        project: str,
        secrets: mlrun.api.schemas.SecretsData,
        allow_internal_secrets: bool = False,
    ):
        if secrets.secrets:
            for secret_key in secrets.secrets.keys():
                mlrun.utils.helpers.verify_field_regex(
                    "secret.key", secret_key, mlrun.utils.regex.secret_key
                )
                if (
                    self._is_internal_secret_key(secret_key)
                    and not allow_internal_secrets
                ):
                    raise mlrun.errors.MLRunAccessDeniedError(
                        f"Not allowed to create/update internal secrets (key starts with "
                        f"{self.internal_secrets_key_prefix})"
                    )
        if secrets.provider == mlrun.api.schemas.SecretProviderName.vault:
            # Init is idempotent and will do nothing if infra is already in place
            mlrun.utils.vault.init_project_vault_configuration(project)

            # If no secrets were passed, no need to touch the actual secrets.
            if secrets.secrets:
                mlrun.utils.vault.store_vault_project_secrets(project, secrets.secrets)
        elif secrets.provider == mlrun.api.schemas.SecretProviderName.kubernetes:
            if mlrun.api.utils.singletons.k8s.get_k8s():
                mlrun.api.utils.singletons.k8s.get_k8s().store_project_secrets(
                    project, secrets.secrets
                )
            else:
                raise mlrun.errors.MLRunInternalServerError(
                    "K8s provider cannot be initialized"
                )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {secrets.provider}"
            )

    def delete_secrets(
        self,
        project: str,
        provider: mlrun.api.schemas.SecretProviderName,
        secrets: typing.Optional[typing.List[str]] = None,
        allow_internal_secrets: bool = False,
    ):
        if not allow_internal_secrets:
            if secrets:
                for secret_key in secrets:
                    if self._is_internal_secret_key(secret_key):
                        raise mlrun.errors.MLRunAccessDeniedError(
                            f"Not allowed to delete internal secrets (key starts with "
                            f"{self.internal_secrets_key_prefix})"
                        )
            else:
                # When secrets are not provided the default behavior will be to delete them all, but if internal secrets
                # are not allowed, we don't want to delete them, so we list the non internal keys
                secrets = self.list_secret_keys(
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

    def list_secret_keys(
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
                filter(lambda key: not self._is_internal_secret_key(key), secret_keys)
            )

        return mlrun.api.schemas.SecretKeysData(
            provider=provider, secret_keys=secret_keys
        )

    def list_secrets(
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
            secrets_data = mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_data(
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
                if not self._is_internal_secret_key(key)
            }
        return mlrun.api.schemas.SecretsData(provider=provider, secrets=secrets_data)

    def _is_internal_secret_key(self, key: str):
        return key.startswith(self.internal_secrets_key_prefix)
