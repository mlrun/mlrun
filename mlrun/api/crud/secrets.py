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
                    secret_key.startswith(self.internal_secrets_key_prefix)
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
    ):
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
    ) -> mlrun.api.schemas.SecretKeysData:
        if provider == mlrun.api.schemas.SecretProviderName.vault:
            if not token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Vault list project secret keys request without providing token"
                )

            vault = mlrun.utils.vault.VaultStore(token)
            secret_values = vault.get_secrets(None, project=project)
            return mlrun.api.schemas.SecretKeysData(
                provider=provider, secret_keys=list(secret_values.keys())
            )
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
                return mlrun.api.schemas.SecretKeysData(
                    provider=provider, secret_keys=secret_keys
                )
            else:
                raise mlrun.errors.MLRunInternalServerError(
                    "K8s provider cannot be initialized"
                )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {provider}"
            )

    def _get_kubernetes_secret_value(
        self, project: str, key: str, raise_on_not_found: bool = True
    ) -> typing.Optional[str]:
        data = mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_data(project)
        if key not in data:
            if raise_on_not_found:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Key not found in secret. key={key}"
                )
            else:
                return None
        return data[key]

    def list_secrets(
        self,
        project: str,
        provider: mlrun.api.schemas.SecretProviderName,
        secrets: typing.Optional[typing.List[str]] = None,
        token: typing.Optional[str] = None,
    ) -> mlrun.api.schemas.SecretsData:
        if provider == mlrun.api.schemas.SecretProviderName.vault:
            if not token:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Vault list project secrets request without providing token"
                )

            vault = mlrun.utils.vault.VaultStore(token)
            secret_values = vault.get_secrets(secrets, project=project)
            return mlrun.api.schemas.SecretsData(
                provider=provider, secrets=secret_values
            )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Provider requested is not supported. provider = {provider}"
            )
