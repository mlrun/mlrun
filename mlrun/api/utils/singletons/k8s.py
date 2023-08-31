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
import time
import typing

from kubernetes import client, config
from kubernetes.client.rest import ApiException

import mlrun.api.runtime_handlers.mpijob
import mlrun.common.schemas
import mlrun.common.secrets
import mlrun.config as mlconfig
import mlrun.errors
import mlrun.platforms.iguazio
import mlrun.runtimes
from mlrun.utils import logger

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


class SecretTypes:
    opaque = "Opaque"
    v3io_fuse = "v3io/fuse"


class K8sHelper(mlrun.common.secrets.SecretProviderInterface):
    def __init__(self, namespace=None, silent=False, log=True):
        self.namespace = namespace or mlconfig.config.namespace
        self.config_file = mlconfig.config.kubernetes.kubeconfig_path or None
        self.running_inside_kubernetes_cluster = False
        try:
            self._init_k8s_config(log)
            self.v1api = client.CoreV1Api()
            self.crdapi = client.CustomObjectsApi()
        except Exception as exc:
            logger.warning(
                "cannot initialize kubernetes client", exc=mlrun.errors.err_to_str(exc)
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
                logger.info("using in-cluster config.")
        except Exception:
            try:
                config.load_kube_config(self.config_file)
                if log:
                    logger.info("using local kubernetes config.")
            except Exception:
                raise RuntimeError(
                    "cannot find local kubernetes config file,"
                    " place it in ~/.kube/config or specify it in "
                    "KUBECONFIG env var"
                )

    def is_running_inside_kubernetes_cluster(self):
        return self.running_inside_kubernetes_cluster

    def list_pods(self, namespace=None, selector="", states=None):
        try:
            resp = self.v1api.list_namespaced_pod(
                self.resolve_namespace(namespace), label_selector=selector
            )
        except ApiException as exc:
            logger.error(f"failed to list pods: {mlrun.errors.err_to_str(exc)}")
            raise exc

        items = []
        for i in resp.items:
            if not states or i.status.phase in states:
                items.append(i)
        return items

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
                        "failed to create pod after max retries",
                        retry_count=retry_count,
                        exc=mlrun.errors.err_to_str(exc),
                        pod=pod,
                    )
                    raise exc

                logger.error(
                    "failed to create pod", exc=mlrun.errors.err_to_str(exc), pod=pod
                )

                # known k8s issue, see https://github.com/kubernetes/kubernetes/issues/67761
                if "gke-resource-quotas" in mlrun.errors.err_to_str(exc):
                    logger.warning(
                        "failed to create pod due to gke resource error, "
                        f"sleeping {retry_interval} seconds and retrying"
                    )
                    retry_count += 1
                    time.sleep(retry_interval)
                    continue

                raise exc
            else:
                logger.info(f"Pod {resp.metadata.name} created")
                return resp.metadata.name, resp.metadata.namespace

    def delete_pod(self, name, namespace=None, grace_period_seconds=0):
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
                    f"failed to delete pod: {mlrun.errors.err_to_str(exc)}",
                    pod_name=name,
                )
                raise exc

    def get_pod(self, name, namespace=None, raise_on_not_found=False):
        try:
            api_response = self.v1api.read_namespaced_pod(
                name=name, namespace=self.resolve_namespace(namespace)
            )
            return api_response
        except ApiException as exc:
            if exc.status != 404:
                logger.error(f"failed to get pod: {mlrun.errors.err_to_str(exc)}")
                raise exc
            else:
                if raise_on_not_found:
                    raise mlrun.errors.MLRunNotFoundError(f"Pod not found: {name}")
            return None

    def get_pod_status(self, name, namespace=None):
        return self.get_pod(
            name, namespace, raise_on_not_found=True
        ).status.phase.lower()

    def delete_crd(self, name, crd_group, crd_version, crd_plural, namespace=None):
        try:
            namespace = self.resolve_namespace(namespace)
            self.crdapi.delete_namespaced_custom_object(
                crd_group,
                crd_version,
                namespace,
                crd_plural,
                name,
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
                    f"failed to delete crd: {mlrun.errors.err_to_str(exc)}",
                    crd_name=name,
                    crd_group=crd_group,
                    crd_version=crd_version,
                    crd_plural=crd_plural,
                )
                raise exc

    def logs(self, name, namespace=None):
        try:
            resp = self.v1api.read_namespaced_pod_log(
                name=name, namespace=self.resolve_namespace(namespace)
            )
        except ApiException as exc:
            logger.error(f"failed to get pod logs: {mlrun.errors.err_to_str(exc)}")
            raise exc

        return resp

    def get_logger_pods(self, project, uid, run_kind, namespace=""):
        namespace = self.resolve_namespace(namespace)
        mpijob_crd_version = (
            mlrun.api.runtime_handlers.mpijob.resolve_mpijob_crd_version()
        )
        mpijob_role_label = (
            mlrun.runtimes.constants.MPIJobCRDVersions.role_label_by_version(
                mpijob_crd_version
            )
        )
        extra_selectors = {
            "spark": "spark-role=driver",
            "mpijob": f"{mpijob_role_label}=launcher",
        }

        # TODO: all mlrun labels are sprinkled in a lot of places - they need to all be defined in a central,
        #  inclusive place.
        selectors = [
            "mlrun/class",
            f"mlrun/project={project}",
            f"mlrun/uid={uid}",
        ]

        # In order to make the `list_pods` request return a lighter and quicker result, we narrow the search for
        # the relevant pods using the proper label selector according to the run kind
        if run_kind in extra_selectors:
            selectors.append(extra_selectors[run_kind])

        selector = ",".join(selectors)
        pods = self.list_pods(namespace, selector=selector)
        if not pods:
            logger.error("no pod matches that uid", uid=uid)
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
                    f"failed to retrieve service accounts: {mlrun.errors.err_to_str(exc)}"
                )
                raise exc
            return None

        if len(service_account.secrets) > 1:
            raise ValueError(
                f"Service account {service_account_name} has more than one secret"
            )

        return service_account.secrets[0].name

    def get_project_secret_name(self, project) -> str:
        return mlconfig.config.secret_stores.kubernetes.project_secret_name.format(
            project=project
        )

    def resolve_auth_secret_name(self, access_key: str) -> str:
        hashed_access_key = self._hash_access_key(access_key)
        return mlconfig.config.secret_stores.kubernetes.auth_secret_name.format(
            hashed_access_key=hashed_access_key
        )

    @staticmethod
    def _hash_access_key(access_key: str):
        return hashlib.sha224(access_key.encode()).hexdigest()

    def store_project_secrets(
        self, project, secrets, namespace=""
    ) -> (str, typing.Optional[mlrun.common.schemas.SecretEventActions]):
        secret_name = self.get_project_secret_name(project)
        action = self.store_secrets(secret_name, secrets, namespace)
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
            labels={"mlrun/username": username},
        )
        return secret_name, action

    def store_secrets(
        self,
        secret_name,
        secrets,
        namespace="",
        type_=SecretTypes.opaque,
        labels: typing.Optional[dict] = None,
    ) -> typing.Optional[mlrun.common.schemas.SecretEventActions]:
        """
        Store secrets in a kubernetes secret object
        :param secret_name: the project secret name
        :param secrets:     the secrets to delete
        :param namespace:   k8s namespace
        :param type_:       k8s secret type
        :param labels:      k8s labels for the secret
        :return: returns the action if the secret was created or updated, None if nothing changed
        """
        namespace = self.resolve_namespace(namespace)
        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret doesn't exist, we'll simply create it
            if exc.status != 404:
                logger.error(
                    f"failed to retrieve k8s secret: {mlrun.errors.err_to_str(exc)}"
                )
                raise exc
            k8s_secret = client.V1Secret(type=type_)
            k8s_secret.metadata = client.V1ObjectMeta(
                name=secret_name, namespace=namespace, labels=labels
            )
            k8s_secret.string_data = secrets
            self.v1api.create_namespaced_secret(namespace, k8s_secret)
            return mlrun.common.schemas.SecretEventActions.created

        secret_data = k8s_secret.data.copy()
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

    def delete_secrets(
        self, secret_name, secrets, namespace=""
    ) -> typing.Optional[mlrun.common.schemas.SecretEventActions]:
        """
        Delete secrets from a kubernetes secret object
        :param secret_name: the project secret name
        :param secrets:     the secrets to delete
        :param namespace:   k8s namespace
        :return: returns the action if the secret was deleted or updated, None if nothing changed
        """
        namespace = self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            if exc.status == 404:
                logger.info(
                    "Project secret does not exist, nothing to delete.",
                    secret_name=secret_name,
                )
                return None
            else:
                logger.error(
                    f"failed to retrieve k8s secret: {mlrun.errors.err_to_str(exc)}"
                )
                raise exc

        secret_data = {}
        if secrets:
            secret_data = k8s_secret.data.copy()
            for secret in secrets:
                secret_data.pop(secret, None)

        if secret_data:
            k8s_secret.data = secret_data
            self.v1api.replace_namespaced_secret(secret_name, namespace, k8s_secret)
            return mlrun.common.schemas.SecretEventActions.updated

        self.v1api.delete_namespaced_secret(secret_name, namespace)
        return mlrun.common.schemas.SecretEventActions.deleted

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
            "mlrun/task-name": task_name,
            "mlrun/class": kind,
            "mlrun/project": self.project,
        }
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

    def set_node_selector(self, node_selector: typing.Optional[typing.Dict[str, str]]):
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
