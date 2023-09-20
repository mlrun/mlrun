import copy
import typing
import uuid
from abc import ABC

from kubernetes import client as k8s_client

import mlrun
import mlrun.api.utils.singletons.k8s
import mlrun.runtimes.pod
import mlrun.secrets
import mlrun.utils.helpers
from mlrun.api.runtime_handlers.base import BaseRuntimeHandler
from mlrun.runtimes.utils import mlrun_key
from mlrun.utils import logger


class KubeResourceHandler(BaseRuntimeHandler, ABC):
    @staticmethod
    def _get_meta(
        runtime: mlrun.runtimes.pod.KubeResource,
        run: mlrun.run.RunObject,
        unique: bool = False,
    ) -> k8s_client.V1ObjectMeta:
        namespace = mlrun.api.utils.singletons.k8s.get_k8s_helper().resolve_namespace()

        labels = get_resource_labels(runtime, run, run.spec.scrape_metrics)
        new_meta = k8s_client.V1ObjectMeta(
            namespace=namespace,
            annotations=runtime.metadata.annotations or run.metadata.annotations,
            labels=labels,
        )

        name = run.metadata.name or "mlrun"
        norm_name = f"{mlrun.utils.helpers.normalize_name(name)}-"
        if unique:
            norm_name += uuid.uuid4().hex[:8]
            new_meta.name = norm_name
            run.set_label("mlrun/job", norm_name)
        else:
            new_meta.generate_name = norm_name
        return new_meta

    def _add_secrets_to_spec_before_running(
        self,
        runtime: mlrun.runtimes.pod.KubeResource,
        run: mlrun.run.RunObject,
        project_name: typing.Optional[str] = None,
    ):
        if runtime._secrets:
            if runtime._secrets.has_vault_source():
                self._add_vault_params_to_spec(
                    runtime=runtime, run=run, project_name=project_name
                )
            if runtime._secrets.has_azure_vault_source():
                self._add_azure_vault_params_to_spec(
                    runtime._secrets.get_azure_vault_k8s_secret()
                )
            self._add_k8s_secrets_to_spec(
                runtime._secrets.get_k8s_secrets(),
                runtime,
                run=run,
                project_name=project_name,
            )
        else:
            self._add_k8s_secrets_to_spec(
                None, runtime, run=run, project_name=project_name
            )

    @staticmethod
    def _add_vault_params_to_spec(
        runtime: mlrun.runtimes.pod.KubeResource,
        run: mlrun.run.RunObject,
        project_name: typing.Optional[str] = None,
    ):
        project_name = project_name or run.metadata.project
        if project_name is None:
            logger.warning("No project provided. Cannot add vault parameters")
            return

        service_account_name = (
            mlrun.mlconf.secret_stores.vault.project_service_account_name.format(
                project=project_name
            )
        )

        project_vault_secret_name = mlrun.api.utils.singletons.k8s.get_k8s_helper().get_project_vault_secret_name(
            project_name, service_account_name
        )
        if project_vault_secret_name is None:
            logger.info(f"No vault secret associated with project {project_name}")
            return

        volumes = [
            {
                "name": "vault-secret",
                "secret": {"defaultMode": 420, "secretName": project_vault_secret_name},
            }
        ]
        # We cannot use expanduser() here, since the user in question is the user running in the pod
        # itself (which is root) and not where this code is running. That's why this hacky replacement is needed.
        token_path = mlrun.mlconf.secret_stores.vault.token_path.replace("~", "/root")

        volume_mounts = [{"name": "vault-secret", "mountPath": token_path}]

        runtime.spec.update_vols_and_mounts(volumes, volume_mounts)
        runtime.spec.env.append(
            {
                "name": "MLRUN_SECRET_STORES__VAULT__ROLE",
                "value": f"project:{project_name}",
            }
        )
        # In case remote URL is different from local URL, use it. Else, use the local URL
        vault_url = mlrun.mlconf.secret_stores.vault.remote_url
        if vault_url == "":
            vault_url = mlrun.mlconf.secret_stores.vault.url

        runtime.spec.env.append(
            {"name": "MLRUN_SECRET_STORES__VAULT__URL", "value": vault_url}
        )

    @staticmethod
    def _add_azure_vault_params_to_spec(
        runtime: mlrun.runtimes.pod.KubeResource,
        k8s_secret_name: typing.Optional[str] = None,
    ):
        secret_name = (
            k8s_secret_name
            or mlrun.mlconf.secret_stores.azure_vault.default_secret_name
        )
        if not secret_name:
            logger.warning(
                "No k8s secret provided. Azure key vault will not be available"
            )
            return

        # We cannot use expanduser() here, since the user in question is the user running in the pod
        # itself (which is root) and not where this code is running. That's why this hacky replacement is needed.
        secret_path = mlrun.mlconf.secret_stores.azure_vault.secret_path.replace(
            "~", "/root"
        )
        volumes = [
            {
                "name": "azure-vault-secret",
                "secret": {"defaultMode": 420, "secretName": secret_name},
            }
        ]
        volume_mounts = [{"name": "azure-vault-secret", "mountPath": secret_path}]
        runtime.spec.update_vols_and_mounts(volumes, volume_mounts)

    @staticmethod
    def _add_k8s_secrets_to_spec(
        secrets,
        runtime: mlrun.runtimes.pod.KubeResource,
        run: mlrun.run.RunObject,
        project_name: typing.Optional[str] = None,
        encode_key_names: bool = True,
    ):
        # Check if we need to add the keys of a global secret. Global secrets are intentionally added before
        # project secrets, to allow project secret keys to override them
        global_secret_name = (
            mlrun.mlconf.secret_stores.kubernetes.global_function_env_secret_name
        )
        if mlrun.config.is_running_as_api() and global_secret_name:
            global_secrets = (
                mlrun.api.utils.singletons.k8s.get_k8s_helper().get_secret_data(
                    global_secret_name
                )
            )
            for key, value in global_secrets.items():
                env_var_name = (
                    mlrun.secrets.SecretsStore.k8s_env_variable_name_for_secret(key)
                    if encode_key_names
                    else key
                )
                runtime.set_env_from_secret(env_var_name, global_secret_name, key)

        # the secrets param may be an empty dictionary (asking for all secrets of that project) -
        # it's a different case than None (not asking for project secrets at all).
        if (
            secrets is None
            and not mlrun.mlconf.secret_stores.kubernetes.auto_add_project_secrets
        ):
            return

        project_name = project_name or run.metadata.project
        if project_name is None:
            logger.warning("No project provided. Cannot add k8s secrets")
            return

        secret_name = (
            mlrun.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
                project_name
            )
        )
        # Not utilizing the same functionality from the Secrets crud object because this code also runs client-side
        # in the nuclio remote-dashboard flow, which causes dependency problems.
        existing_secret_keys = (
            mlrun.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_keys(
                project_name, filter_internal=True
            )
        )

        # If no secrets were passed or auto-adding all secrets, we need all existing keys
        if not secrets:
            secrets = {
                key: mlrun.secrets.SecretsStore.k8s_env_variable_name_for_secret(key)
                if encode_key_names
                else key
                for key in existing_secret_keys
            }

        for key, env_var_name in secrets.items():
            if key in existing_secret_keys:
                runtime.set_env_from_secret(env_var_name, secret_name, key)

        # Keep a list of the variables that relate to secrets, so that the MLRun context (when using nuclio:mlrun)
        # can be initialized with those env variables as secrets
        if not encode_key_names and secrets.keys():
            runtime.set_env("MLRUN_PROJECT_SECRETS_LIST", ",".join(secrets.keys()))


def get_resource_labels(function, run=None, scrape_metrics=None):
    scrape_metrics = (
        scrape_metrics if scrape_metrics is not None else mlrun.mlconf.scrape_metrics
    )
    run_uid, run_name, run_project, run_owner = None, None, None, None
    if run:
        run_uid = run.metadata.uid
        run_name = run.metadata.name
        run_project = run.metadata.project
        run_owner = run.metadata.labels.get("owner")
    labels = copy.deepcopy(function.metadata.labels)
    labels[mlrun_key + "class"] = function.kind
    labels[mlrun_key + "project"] = run_project or function.metadata.project
    labels[mlrun_key + "function"] = str(function.metadata.name)
    labels[mlrun_key + "tag"] = str(function.metadata.tag or "latest")
    labels[mlrun_key + "scrape-metrics"] = str(scrape_metrics)

    if run_uid:
        labels[mlrun_key + "uid"] = run_uid

    if run_name:
        labels[mlrun_key + "name"] = run_name

    if run_owner:
        labels[mlrun_key + "owner"] = run_owner

    return labels
