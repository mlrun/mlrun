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
import base64
import hashlib
import random
import string
import time
import typing

from kubernetes import client, config
from kubernetes.client.rest import ApiException

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.common.secrets as mlsecrets
import mlrun.errors
import mlrun.platforms.iguazio
import mlrun.runtimes
import mlrun.runtimes.pod
import server.api.runtime_handlers
from mlrun.utils import logger
from mlrun.utils.helpers import to_non_empty_values_dict

_k8s = None


def get_k8s_helper(namespace=None, silent=True, log=False) -> "K8sHelper":
    """
    Get a k8s helper singleton object
    :param namespace: the namespace to use, if not specified will use the namespace configured in mlrun config
    :param silent: set to true if you're calling this function from a code that might run from remotely (outside of a
    k8s cluster)
    :param log: sometimes we want to avoid logging when executing init_k8s_config
    """
    global _k8s
    if not _k8s:
        _k8s = K8sHelper(namespace, silent=silent, log=log)
    return _k8s


def raise_for_status_code(func):
    """
    A decorator for calls to k8s api when no error handling is needed.
    Raises the matching mlrun exception to the status code.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApiException as exc:
            raise mlrun.errors.err_for_status_code(
                exc.status, message=mlrun.errors.err_to_str(exc)
            ) from exc

    return wrapper


class SecretTypes:
    opaque = "Opaque"
    v3io_fuse = "v3io/fuse"


class K8sHelper(mlsecrets.SecretProviderInterface):
    def __init__(self, namespace=None, silent=False, log=True):
        self.namespace = namespace or mlrun.mlconf.namespace
        self.config_file = mlrun.mlconf.kubernetes.kubeconfig_path or None
        self.running_inside_kubernetes_cluster = False
        try:
            self._init_k8s_config(log)
            self.v1api = client.CoreV1Api()
            self.crdapi = client.CustomObjectsApi()
        except Exception as exc:
            logger.warning(
                "Cannot initialize kubernetes client", exc=mlrun.errors.err_to_str(exc)
            )
            if not silent:
                raise

    def resolve_namespace(self, namespace=None):
        return namespace or self.namespace

    def _init_k8s_config(self, log=True):
        try:
            config.load_incluster_config()
            self.running_inside_kubernetes_cluster = True
            if log:
                logger.info("Using in-cluster config.")
        except Exception:
            try:
                config.load_kube_config(self.config_file)
                self.running_inside_kubernetes_cluster = True
                if log:
                    logger.info("Using local kubernetes config.")
            except Exception:
                raise RuntimeError(
                    "Cannot find local kubernetes config file,"
                    " place it in ~/.kube/config or specify it in "
                    "KUBECONFIG env var"
                )

    def is_running_inside_kubernetes_cluster(self):
        return self.running_inside_kubernetes_cluster

    @raise_for_status_code
    def list_pods(self, namespace=None, selector="", states=None):
        resp = self.v1api.list_namespaced_pod(
            self.resolve_namespace(namespace), label_selector=selector
        )
        items = []
        for i in resp.items:
            if not states or i.status.phase in states:
                items.append(i)
        return items

    @raise_for_status_code
    def list_pods_paginated(
        self,
        namespace: str = None,
        selector: str = "",
        states: list[str] = None,
        max_retry: int = 3,
    ):
        """
        List pods paginated
        :param namespace:       Namespace to query
        :param selector:        Pods label selector
        :param states:          List of pod states to filter by
        :param max_retry:       Maximum number of retries on 410 Gone (when continue token is stale)
        """
        _continue = None
        retry_count = 0
        limit = int(mlrun.mlconf.kubernetes.pagination.list_pods_limit)
        if limit <= 0:
            limit = None
        while True:
            try:
                pods_list = self.v1api.list_namespaced_pod(
                    self.resolve_namespace(namespace),
                    label_selector=selector,
                    watch=False,
                    limit=limit,
                    _continue=_continue,
                )
            except ApiException as exc:
                self._validate_paginated_list_retry(
                    exc, retry_count, max_retry, resource_name="pods"
                )
                _continue = None
                retry_count += 1
                continue

            for item in pods_list.items:
                if not states or item.status.phase in states:
                    yield item

            _continue = pods_list.metadata._continue

            if not _continue:
                break

    @raise_for_status_code
    def list_crds_paginated(
        self,
        crd_group: str,
        crd_version: str,
        crd_plural: str,
        namespace: str = None,
        selector: str = "",
        max_retry: int = 3,
    ):
        """
        List custom resources paginated
        :param crd_group:       The CRD group name
        :param crd_version:     The CRD version
        :param crd_plural:      The CRD plural name
        :param namespace:       Namespace to query
        :param selector:        Custom resource's label selector
        :param max_retry:       Maximum number of retries on 410 Gone (when continue token is stale)
        """
        _continue = None
        retry_count = 0
        limit = int(mlrun.mlconf.kubernetes.pagination.list_crd_objects_limit)
        if limit <= 0:
            limit = None
        while True:
            crd_objects = {}
            crd_items = []
            try:
                crd_objects = self.crdapi.list_namespaced_custom_object(
                    crd_group,
                    crd_version,
                    self.resolve_namespace(namespace),
                    crd_plural,
                    label_selector=selector,
                    limit=limit,
                    _continue=_continue,
                    watch=False,
                )
            except ApiException as exc:
                # ignore error if crd is not defined
                if exc.status != 404:
                    self._validate_paginated_list_retry(
                        exc, retry_count, max_retry, resource_name=crd_plural
                    )
                    _continue = None
                    retry_count += 1
                    continue

            else:
                crd_items = crd_objects["items"]

            yield from crd_items

            _continue = crd_objects["metadata"]["continue"] if crd_objects else None

            if not _continue:
                break

    def create_pod(self, pod, max_retry=3, retry_interval=3):
        if "pod" in dir(pod):
            pod = pod.pod
        pod.metadata.namespace = self.resolve_namespace(pod.metadata.namespace)

        retry_count = 0
        while True:
            try:
                resp = self.v1api.create_namespaced_pod(pod.metadata.namespace, pod)
            except ApiException as exc:
                if retry_count > max_retry:
                    logger.error(
                        "Failed to create pod after max retries",
                        retry_count=retry_count,
                        exc=mlrun.errors.err_to_str(exc),
                        pod=pod,
                    )
                    raise mlrun.errors.err_for_status_code(
                        exc.status, message=mlrun.errors.err_to_str(exc)
                    ) from exc

                logger.error(
                    "Failed to create pod", exc=mlrun.errors.err_to_str(exc), pod=pod
                )

                # known k8s issue, see https://github.com/kubernetes/kubernetes/issues/67761
                if "gke-resource-quotas" in mlrun.errors.err_to_str(exc):
                    logger.warning(
                        "Failed to create pod due to gke resource error, sleeping and retrying",
                        retry_interval=retry_interval,
                    )
                    retry_count += 1
                    time.sleep(retry_interval)
                    continue

                raise mlrun.errors.err_for_status_code(
                    exc.status, message=mlrun.errors.err_to_str(exc)
                ) from exc
            else:
                logger.info("Pod created", pod_name=resp.metadata.name)
                return resp.metadata.name, resp.metadata.namespace

    def delete_pod(self, name, namespace=None, grace_period_seconds=None):
        try:
            api_response = self.v1api.delete_namespaced_pod(
                name,
                self.resolve_namespace(namespace),
                grace_period_seconds=grace_period_seconds,
                propagation_policy="Background",
            )
            return api_response
        except ApiException as exc:
            # ignore error if pod is already removed
            if exc.status != 404:
                logger.error(
                    "Failed to delete pod",
                    pod_name=name,
                    exc=mlrun.errors.err_to_str(exc),
                )
                raise mlrun.errors.err_for_status_code(
                    exc.status, message=mlrun.errors.err_to_str(exc)
                ) from exc

    def get_pod(self, name, namespace=None, raise_on_not_found=False):
        try:
            api_response = self.v1api.read_namespaced_pod(
                name=name, namespace=self.resolve_namespace(namespace)
            )
            return api_response
        except ApiException as exc:
            if exc.status != 404:
                logger.error(
                    "Failed to get pod", pod_name=name, exc=mlrun.errors.err_to_str(exc)
                )
                raise mlrun.errors.err_for_status_code(
                    exc.status, message=mlrun.errors.err_to_str(exc)
                ) from exc
            else:
                if raise_on_not_found:
                    raise mlrun.errors.MLRunNotFoundError(f"Pod not found: {name}")
            return None

    def get_pod_status(self, name, namespace=None):
        return self.get_pod(
            name, namespace, raise_on_not_found=True
        ).status.phase.lower()

    def delete_crd(
        self,
        name,
        crd_group,
        crd_version,
        crd_plural,
        namespace=None,
        grace_period_seconds=None,
    ):
        try:
            namespace = self.resolve_namespace(namespace)
            self.crdapi.delete_namespaced_custom_object(
                crd_group,
                crd_version,
                namespace,
                crd_plural,
                name,
                grace_period_seconds=grace_period_seconds,
            )
            logger.info(
                "Deleted crd object",
                crd_name=name,
                namespace=namespace,
            )
        except ApiException as exc:
            # ignore error if crd is already removed
            if exc.status != 404:
                logger.error(
                    "Failed to delete crd object",
                    exc=mlrun.errors.err_to_str(exc),
                    crd_name=name,
                    crd_group=crd_group,
                    crd_version=crd_version,
                    crd_plural=crd_plural,
                )
                raise mlrun.errors.err_for_status_code(
                    exc.status, message=mlrun.errors.err_to_str(exc)
                ) from exc

    def logs(self, name, namespace=None):
        try:
            resp = self.v1api.read_namespaced_pod_log(
                name=name, namespace=self.resolve_namespace(namespace)
            )
        except ApiException as exc:
            logger.error("Failed to get pod logs", exc=mlrun.errors.err_to_str(exc))
            raise exc

        return resp

    def get_logger_pods(self, project, uid, run_kind, namespace=""):
        namespace = self.resolve_namespace(namespace)
        mpijob_crd_version = server.api.runtime_handlers.resolve_mpijob_crd_version()
        mpijob_role_label = (
            mlrun.common.runtimes.constants.MPIJobCRDVersions.role_label_by_version(
                mpijob_crd_version
            )
        )
        extra_selectors = {
            "spark": f"{mlrun_constants.MLRunInternalLabels.spark_role}=driver",
            "mpijob": f"{mpijob_role_label}=launcher",
        }

        selectors = [
            mlrun_constants.MLRunInternalLabels.mlrun_class,
            f"{mlrun_constants.MLRunInternalLabels.project}={project}",
            f"{mlrun_constants.MLRunInternalLabels.uid}={uid}",
        ]

        # In order to make the `list_pods` request return a lighter and quicker result, we narrow the search for
        # the relevant pods using the proper label selector according to the run kind
        if run_kind in extra_selectors:
            selectors.append(extra_selectors[run_kind])

        selector = ",".join(selectors)
        pods = self.list_pods(namespace, selector=selector)
        if not pods:
            logger.error("No pod matches that uid", uid=uid)
            return

        return {p.metadata.name: p.status.phase for p in pods}

    def get_project_vault_secret_name(
        self, project, service_account_name, namespace=""
    ):
        namespace = self.resolve_namespace(namespace)

        try:
            service_account = self.v1api.read_namespaced_service_account(
                service_account_name, namespace
            )
        except ApiException as exc:
            # It's valid for the service account to not exist. Simply return None
            if exc.status != 404:
                logger.error(
                    "Failed to retrieve service accounts",
                    service_account_name=service_account_name,
                    exc=mlrun.errors.err_to_str(exc),
                )
                raise mlrun.errors.err_for_status_code(
                    exc.status, message=mlrun.errors.err_to_str(exc)
                ) from exc
            return None

        if len(service_account.secrets) > 1:
            raise ValueError(
                f"Service account {service_account_name} has more than one secret"
            )

        return service_account.secrets[0].name

    def get_project_secret_name(self, project) -> str:
        return mlrun.mlconf.secret_stores.kubernetes.project_secret_name.format(
            project=project
        )

    def resolve_auth_secret_name(self, access_key: str) -> str:
        hashed_access_key = self._hash_access_key(access_key)
        return mlrun.mlconf.secret_stores.kubernetes.auth_secret_name.format(
            hashed_access_key=hashed_access_key
        )

    def store_project_secrets(
        self, project, secrets, namespace=""
    ) -> (str, typing.Optional[mlrun.common.schemas.SecretEventActions]):
        secret_name = self.get_project_secret_name(project)
        action = self.store_secrets(
            secret_name, secrets, namespace, retry_on_conflict=True
        )
        return secret_name, action

    def read_auth_secret(self, secret_name, namespace="", raise_on_not_found=False):
        namespace = self.resolve_namespace(namespace)

        try:
            secret_data = self.v1api.read_namespaced_secret(secret_name, namespace).data
        except ApiException as exc:
            logger.error(
                "Failed to read secret",
                secret_name=secret_name,
                namespace=namespace,
                exc=mlrun.errors.err_to_str(exc),
            )
            if exc.status != 404:
                raise exc
            elif raise_on_not_found:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Secret '{secret_name}' was not found in namespace '{namespace}'"
                ) from exc

            return None, None

        def _get_secret_value(key):
            if secret_data.get(key):
                return base64.b64decode(secret_data[key]).decode("utf-8")
            else:
                return None

        username = _get_secret_value(
            mlrun.common.schemas.AuthSecretData.get_field_secret_key("username")
        )
        access_key = _get_secret_value(
            mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key")
        )

        return username, access_key

    def store_auth_secret(
        self, username: str, access_key: str, namespace=""
    ) -> (str, typing.Optional[mlrun.common.schemas.SecretEventActions]):
        """
        Store the given access key as a secret in the cluster. The secret name is generated from the access key
        :return: returns the secret name and the action taken against the secret
        """
        secret_name = self.resolve_auth_secret_name(access_key)
        secret_data = {
            mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                "username"
            ): username,
            mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                "access_key"
            ): access_key,
        }
        action = self.store_secrets(
            secret_name,
            secret_data,
            namespace,
            type_=SecretTypes.v3io_fuse,
            labels=self._resolve_secret_labels(username),
            retry_on_conflict=True,
        )
        return secret_name, action

    @raise_for_status_code
    def store_secrets(
        self,
        secret_name,
        secrets,
        namespace="",
        type_=SecretTypes.opaque,
        labels: typing.Optional[dict] = None,
        retry_on_conflict: bool = False,
    ) -> typing.Optional[mlrun.common.schemas.SecretEventActions]:
        """
        Store secrets in a kubernetes secret object
        :param secret_name: the project secret name
        :param secrets:     the secrets to create
        :param namespace:   k8s namespace
        :param type_:       k8s secret type
        :param labels:      k8s labels for the secret
        :param retry_on_conflict:   if True, will retry to create the secret for race conditions
        :return: returns the action if the secret was created or updated, None if nothing changed
        """
        if not secrets:
            # Nothing to store
            return

        namespace = self.resolve_namespace(namespace)
        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret doesn't exist, we'll simply create it
            if exc.status != 404:
                logger.error(
                    "Failed to retrieve k8s secret",
                    secret_name=secret_name,
                    exc=mlrun.errors.err_to_str(exc),
                )
                raise exc
            k8s_secret = client.V1Secret(type=type_)
            k8s_secret.metadata = client.V1ObjectMeta(
                name=secret_name, namespace=namespace, labels=labels
            )
            k8s_secret.string_data = secrets
            try:
                self.v1api.create_namespaced_secret(namespace, k8s_secret)
                return mlrun.common.schemas.SecretEventActions.created
            except ApiException as exc:
                if exc.status == 409 and retry_on_conflict:
                    logger.warning(
                        "Secret was created while we tried to create it, retrying...",
                        secret_name=secret_name,
                        exc=mlrun.errors.err_to_str(exc),
                    )
                    return self.store_secrets(
                        secret_name,
                        secrets,
                        namespace,
                        type_,
                        labels,
                        retry_on_conflict=False,
                    )
                raise exc

        secret_data = k8s_secret.data.copy() if k8s_secret.data else {}

        for key, value in secrets.items():
            secret_data[key] = base64.b64encode(value.encode()).decode("utf-8")

        k8s_secret.data = secret_data
        self.v1api.replace_namespaced_secret(secret_name, namespace, k8s_secret)
        return mlrun.common.schemas.SecretEventActions.updated

    def load_secret(self, secret_name, namespace=""):
        namespace = namespace or self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException:
            return None

        return k8s_secret.data

    def delete_project_secrets(
        self, project, secrets, namespace=""
    ) -> (str, typing.Optional[mlrun.common.schemas.SecretEventActions]):
        """
        Delete secrets from a kubernetes secret object
        :return: returns the secret name and the action taken against the secret
        """
        secret_name = self.get_project_secret_name(project)
        action = self.delete_secrets(secret_name, secrets, namespace)
        return secret_name, action

    def delete_auth_secret(self, secret_ref: str, namespace=""):
        self.delete_secrets(secret_ref, {}, namespace)

    @raise_for_status_code
    def delete_secrets(
        self, secret_name, secrets, namespace=""
    ) -> typing.Optional[mlrun.common.schemas.SecretEventActions]:
        """
        Delete secrets from a kubernetes secret object
        :param secret_name: the project secret name
        :param secrets:     the secrets to delete. If None, all secrets will be deleted
        :param namespace:   k8s namespace
        :return: returns the action if the secret was deleted or updated, None if nothing changed
        """
        namespace = self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            if exc.status == 404:
                logger.info(
                    "Project secret does not exist, nothing to delete",
                    secret_name=secret_name,
                )
                return None
            else:
                logger.error(
                    "Failed to retrieve k8s secret",
                    secret_name=secret_name,
                    exc=mlrun.errors.err_to_str(exc),
                )
                raise exc

        if not k8s_secret.data:
            logger.debug(
                "No data found in the Kubernetes secret",
                secret_name=secret_name,
            )
            self.v1api.delete_namespaced_secret(secret_name, namespace)
            return mlrun.common.schemas.SecretEventActions.deleted

        # Create a copy of the k8s secret data, filtering out specified secrets if any
        if secrets:
            secret_data = {
                key: value
                for key, value in k8s_secret.data.items()
                if key not in secrets
            }
        elif secrets is None:
            # Delete all secrets
            secret_data = {}
        else:
            secret_data = k8s_secret.data.copy()

        # Check if there were any changes to the secret data
        if len(secret_data) == len(k8s_secret.data):
            # No secrets were deleted
            return None

        if secret_data:
            # Update the existing secret with modified data
            k8s_secret.data = secret_data
            self.v1api.replace_namespaced_secret(secret_name, namespace, k8s_secret)
            return mlrun.common.schemas.SecretEventActions.updated

        # No secrets left, so delete the secret
        self.v1api.delete_namespaced_secret(secret_name, namespace)
        return mlrun.common.schemas.SecretEventActions.deleted

    @raise_for_status_code
    def ensure_configmap(
        self,
        resource: str,
        resource_name: str,
        data: dict,
        namespace: str = "",
        labels: dict = None,
        project: str = None,
    ):
        namespace = self.resolve_namespace(namespace)
        have_confmap = False
        label_name = mlrun_constants.MLRunInternalLabels.resource_name
        labels = labels or {}
        labels[label_name] = resource_name
        labels[mlrun_constants.MLRunInternalLabels.project] = project

        configmap_with_label = self.get_configmap(resource_name, namespace)
        if configmap_with_label:
            configmap_name = configmap_with_label.metadata.name
            have_confmap = True
        else:
            full_name = f"{resource}-{resource_name}"
            configmap_name = (
                full_name
                if len(full_name) <= 63
                else full_name[:59] + self._generate_rand_string(4)
            )

        body = client.V1ConfigMap(
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=configmap_name, labels=labels),
            data=data,
        )

        if have_confmap:
            try:
                self.v1api.replace_namespaced_config_map(
                    configmap_name, namespace=namespace, body=body
                )
            except ApiException as exc:
                logger.error(
                    "Failed to replace k8s config map",
                    name=configmap_name,
                    exc=mlrun.errors.err_to_str(exc),
                )
                raise exc
        else:
            try:
                self.v1api.create_namespaced_config_map(namespace=namespace, body=body)
            except ApiException as exc:
                logger.error(
                    "Failed to create k8s config map",
                    name=configmap_name,
                    exc=mlrun.errors.err_to_str(exc),
                )
                raise exc
        return configmap_name

    @raise_for_status_code
    def get_configmap(self, name: str, namespace: str = ""):
        namespace = self.resolve_namespace(namespace)
        label_name = mlrun_constants.MLRunInternalLabels.resource_name
        configmaps_with_label = self.v1api.list_namespaced_config_map(
            namespace=namespace, label_selector=f"{label_name}={name}"
        )
        if len(configmaps_with_label.items) > 1:
            raise mlrun.errors.MLRunInternalServerError(
                f"Received more than one config map for label: {name}"
            )

        return configmaps_with_label.items[0] if configmaps_with_label.items else None

    @raise_for_status_code
    def delete_configmap(self, name: str, namespace: str = "", raise_on_error=True):
        namespace = self.resolve_namespace(namespace)

        try:
            self.v1api.delete_namespaced_config_map(
                name=name,
                namespace=namespace,
            )
        except ApiException as exc:
            logger.error(
                "Failed to delete k8s config map",
                name=name,
                exc=mlrun.errors.err_to_str(exc),
            )
            if raise_on_error:
                raise exc

    @staticmethod
    def _hash_access_key(access_key: str):
        return hashlib.sha224(access_key.encode()).hexdigest()

    @staticmethod
    @raise_for_status_code
    def _validate_paginated_list_retry(
        exc: ApiException, retry_count: int, max_retry: int, resource_name: str
    ):
        """
        Validates 410 Gone retries.
        Raises `exc` if error is not 410 or retries are exhausted.
        Otherwise, logs an appropriate warning.
        :param exc:             The ApiException raised by the list query
        :param retry_count:     The current retry count
        :param max_retry:       The maximum retries allowed
        :param resource_name:   The resource that was listed
        """
        if exc.status != 410:
            raise exc

        if retry_count >= max_retry:
            logger.error(
                "Failed to list resources paginated, max retries exceeded",
                retry_count=retry_count,
                max_retry=max_retry,
                resource_name=resource_name,
            )
            raise exc

        logger.warning(
            "Failed to list resources due to stale continue token. Retrying from scratch",
            retry_count=retry_count,
            resource_name=resource_name,
            exc=mlrun.errors.err_to_str(exc),
        )

    def _get_project_secrets_raw_data(self, project, namespace=""):
        secret_name = self.get_project_secret_name(project)
        return self._get_secret_raw_data(secret_name, namespace)

    def _get_secret_raw_data(self, secret_name, namespace=""):
        namespace = self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException:
            return None

        return k8s_secret.data

    def get_project_secret_keys(self, project, namespace="", filter_internal=False):
        secrets_data = self._get_project_secrets_raw_data(project, namespace)
        if not secrets_data:
            return []

        secret_keys = list(secrets_data.keys())
        if filter_internal:
            secret_keys = list(
                filter(lambda key: not key.startswith("mlrun."), secret_keys)
            )
        return secret_keys

    def get_project_secret_data(self, project, secret_keys=None, namespace=""):
        secrets_data = self._get_project_secrets_raw_data(project, namespace)
        return self._decode_secret_data(secrets_data, secret_keys)

    def get_secret_data(self, secret_name, namespace=""):
        secrets_data = self._get_secret_raw_data(secret_name, namespace)
        return self._decode_secret_data(secrets_data)

    def _decode_secret_data(self, secrets_data, secret_keys=None):
        results = {}
        if not secrets_data:
            return results

        # If not asking for specific keys, return all
        secret_keys = secret_keys or secrets_data.keys()

        for key in secret_keys:
            encoded_value = secrets_data.get(key)
            if encoded_value:
                results[key] = base64.b64decode(secrets_data[key]).decode("utf-8")
        return results

    def _resolve_secret_labels(self, username):
        if not username:
            return {}
        labels = {
            mlrun_constants.MLRunInternalLabels.username: username,
        }
        if "@" in username:
            username, domain = username.split("@")
            labels[mlrun_constants.MLRunInternalLabels.username] = username
            labels[mlrun_constants.MLRunInternalLabels.username_domain] = domain
        return labels

    @staticmethod
    def _generate_rand_string(length):
        return "".join(
            random.choice(string.ascii_lowercase + string.digits) for _ in range(length)
        )


class BasePod:
    def __init__(
        self,
        task_name="",
        image=None,
        command=None,
        args=None,
        namespace="",
        kind="job",
        project=None,
        default_pod_spec_attributes=None,
        resources=None,
        labels=None,
    ):
        self.namespace = namespace
        self.name = ""
        self.task_name = task_name
        self.image = image
        self.command = command
        self.args = args
        self._volumes = []
        self._mounts = []
        self.env = None
        self.node_selector = None
        self.project = project or mlrun.mlconf.default_project
        self._labels = {
            mlrun_constants.MLRunInternalLabels.task_name: task_name,
            mlrun_constants.MLRunInternalLabels.mlrun_class: kind,
            mlrun_constants.MLRunInternalLabels.project: self.project,
        } | (labels or {})
        self._annotations = {}
        self._init_containers = []
        # will be applied on the pod spec only when calling .pod(), allows to override spec attributes
        self.default_pod_spec_attributes = default_pod_spec_attributes
        self.resources = resources

    @property
    def pod(self):
        return self._get_spec()

    @property
    def init_containers(self):
        return self._init_containers

    @init_containers.setter
    def init_containers(self, containers):
        self._init_containers = containers

    def append_init_container(
        self,
        image,
        command=None,
        args=None,
        env=None,
        image_pull_policy="IfNotPresent",
        name="init",
    ):
        if isinstance(env, dict):
            env = [client.V1EnvVar(name=k, value=v) for k, v in env.items()]
        self._init_containers.append(
            client.V1Container(
                name=name,
                image=image,
                env=env,
                command=command,
                args=args,
                image_pull_policy=image_pull_policy,
            )
        )

    def add_label(self, key, value):
        self._labels[key] = str(value)

    def add_annotation(self, key, value):
        self._annotations[key] = str(value)

    def add_volume(self, volume: client.V1Volume, mount_path, name=None, sub_path=None):
        self._mounts.append(
            client.V1VolumeMount(
                name=name or volume.name, mount_path=mount_path, sub_path=sub_path
            )
        )
        self._volumes.append(volume)

    def mount_empty(self, name="empty", mount_path="/empty"):
        self.add_volume(
            client.V1Volume(name=name, empty_dir=client.V1EmptyDirVolumeSource()),
            mount_path=mount_path,
        )

    def mount_v3io(
        self, name="v3io", remote="~/", mount_path="/User", access_key="", user=""
    ):
        self.add_volume(
            mlrun.platforms.iguazio.v3io_to_vol(name, remote, access_key, user),
            mount_path=mount_path,
            name=name,
        )

    def mount_cfgmap(self, name, path="/config"):
        self.add_volume(
            client.V1Volume(
                name=name, config_map=client.V1ConfigMapVolumeSource(name=name)
            ),
            mount_path=path,
        )

    def mount_secret(self, name, path="/secret", items=None, sub_path=None):
        self.add_volume(
            client.V1Volume(
                name=name,
                secret=client.V1SecretVolumeSource(
                    secret_name=name,
                    items=items,
                ),
            ),
            mount_path=path,
            sub_path=sub_path,
        )

    def set_node_selector(self, node_selector: typing.Optional[dict[str, str]]):
        self.node_selector = node_selector

    def _get_spec(self, template=False):
        pod_obj = client.V1PodTemplate if template else client.V1Pod

        if self.env and isinstance(self.env, dict):
            env = [client.V1EnvVar(name=k, value=v) for k, v in self.env.items()]
        else:
            env = self.env
        container = client.V1Container(
            name="base",
            image=self.image,
            env=env,
            command=self.command,
            args=self.args,
            volume_mounts=self._mounts,
            resources=self.resources,
        )

        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            volumes=self._volumes,
            node_selector=self.node_selector,
        )

        # if attribute isn't defined use default pod spec attributes
        for key, val in self.default_pod_spec_attributes.items():
            if not getattr(pod_spec, key, None):
                setattr(pod_spec, key, val)

        for init_containers in self._init_containers:
            init_containers.volume_mounts = self._mounts
        pod_spec.init_containers = self._init_containers

        pod = pod_obj(
            metadata=client.V1ObjectMeta(
                generate_name=f"{self.task_name}-",
                namespace=self.namespace,
                labels=self._labels,
                annotations=self._annotations,
            ),
            spec=pod_spec,
        )
        return pod


def kube_resource_spec_to_pod_spec(
    kube_resource_spec: mlrun.runtimes.pod.KubeResourceSpec,
    container: client.V1Container,
    node_selector: dict = None,
):
    return client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        volumes=kube_resource_spec.volumes,
        service_account=kube_resource_spec.service_account,
        node_name=kube_resource_spec.node_name,
        node_selector=resolve_node_selector(
            node_selector, kube_resource_spec.node_selector
        ),
        affinity=kube_resource_spec.affinity,
        priority_class_name=kube_resource_spec.priority_class_name
        if len(mlrun.mlconf.get_valid_function_priority_class_names())
        else None,
        tolerations=kube_resource_spec.tolerations,
        security_context=kube_resource_spec.security_context,
        termination_grace_period_seconds=kube_resource_spec.termination_grace_period_seconds,
    )


def resolve_node_selector(run_node_selector, runtime_node_selector):
    # To maintain backwards compatibility, use the node_selector from the run object if it exists.
    # otherwise, use the node_selector from the function object.
    node_selector = run_node_selector or runtime_node_selector

    # Ignore empty labels
    return to_non_empty_values_dict(node_selector)
