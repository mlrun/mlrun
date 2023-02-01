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
import base64
import hashlib
import time
import typing
from datetime import datetime
from sys import stdout

import kubernetes.client
from kubernetes import client, config
from kubernetes.client.rest import ApiException

import mlrun.api.schemas
import mlrun.errors

from .config import config as mlconfig
from .errors import err_to_str
from .platforms.iguazio import v3io_to_vol
from .utils import logger

_k8s = None


def get_k8s_helper(namespace=None, silent=False, log=False) -> "K8sHelper":
    """
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


class K8sHelper:
    def __init__(self, namespace=None, config_file=None, silent=False, log=True):
        self.namespace = namespace or mlconfig.namespace
        self.config_file = config_file
        self.running_inside_kubernetes_cluster = False
        try:
            self._init_k8s_config(log)
            self.v1api = client.CoreV1Api()
            self.crdapi = client.CustomObjectsApi()
        except Exception:
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
            logger.error(f"failed to list pods: {err_to_str(exc)}")
            raise exc

        items = []
        for i in resp.items:
            if not states or i.status.phase in states:
                items.append(i)
        return items

    def clean_pods(self, namespace=None, selector="", states=None):
        if not selector and not states:
            raise ValueError("labels selector or states list must be specified")
        items = self.list_pods(namespace, selector, states)
        for item in items:
            self.delete_pod(item.metadata.name, item.metadata.namespace)

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
                        exc=err_to_str(exc),
                        pod=pod,
                    )
                    raise exc

                logger.error("failed to create pod", exc=err_to_str(exc), pod=pod)

                # known k8s issue, see https://github.com/kubernetes/kubernetes/issues/67761
                if "gke-resource-quotas" in err_to_str(exc):
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

    def delete_pod(self, name, namespace=None):
        try:
            api_response = self.v1api.delete_namespaced_pod(
                name,
                self.resolve_namespace(namespace),
                grace_period_seconds=0,
                propagation_policy="Background",
            )
            return api_response
        except ApiException as exc:
            # ignore error if pod is already removed
            if exc.status != 404:
                logger.error(f"failed to delete pod: {err_to_str(exc)}", pod_name=name)
                raise exc

    def get_pod(self, name, namespace=None, raise_on_not_found=False):
        try:
            api_response = self.v1api.read_namespaced_pod(
                name=name, namespace=self.resolve_namespace(namespace)
            )
            return api_response
        except ApiException as exc:
            if exc.status != 404:
                logger.error(f"failed to get pod: {err_to_str(exc)}")
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
                    f"failed to delete crd: {err_to_str(exc)}",
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
            logger.error(f"failed to get pod logs: {err_to_str(exc)}")
            raise exc

        return resp

    def run_job(self, pod, timeout=600):
        pod_name, namespace = self.create_pod(pod)
        if not pod_name:
            logger.error("failed to create pod")
            return "error"
        return self.watch(pod_name, namespace, timeout)

    def watch(self, pod_name, namespace=None, timeout=600, writer=None):
        namespace = self.resolve_namespace(namespace)
        start_time = datetime.now()
        while True:
            try:
                pod = self.get_pod(pod_name, namespace)
                if not pod:
                    return "error"
                status = pod.status.phase.lower()
                if status in ["running", "completed", "succeeded"]:
                    print("")
                    break
                if status == "failed":
                    return "failed"
                elapsed_time = (datetime.now() - start_time).seconds
                if elapsed_time > timeout:
                    return "timeout"
                time.sleep(2)
                stdout.write(".")
                if status != "pending":
                    logger.warning(f"pod state in loop is {status}")
            except ApiException as exc:
                logger.error(f"failed waiting for pod: {err_to_str(exc)}\n")
                return "error"
        outputs = self.v1api.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, follow=True, _preload_content=False
        )
        for out in outputs:
            print(out.decode("utf-8"), end="")
            if writer:
                writer.write(out)

        for i in range(5):
            pod_state = self.get_pod(pod_name, namespace).status.phase.lower()
            if pod_state != "running":
                break
            logger.warning("pod still running, waiting 2 sec")
            time.sleep(2)

        if pod_state == "failed":
            logger.error("pod exited with error")
        if writer:
            writer.flush()
        return pod_state

    def create_cfgmap(self, name, data, namespace="", labels=None):
        body = client.api_client.V1ConfigMap()
        namespace = self.resolve_namespace(namespace)
        body.data = data
        if name.endswith("*"):
            body.metadata = client.V1ObjectMeta(
                generate_name=name[:-1], namespace=namespace, labels=labels
            )
        else:
            body.metadata = client.V1ObjectMeta(
                name=name, namespace=namespace, labels=labels
            )
        try:
            resp = self.v1api.create_namespaced_config_map(namespace, body)
        except ApiException as exc:
            logger.error(f"failed to create configmap: {err_to_str(exc)}")
            raise exc

        logger.info(f"ConfigMap {resp.metadata.name} created")
        return resp.metadata.name

    def del_cfgmap(self, name, namespace=None):
        try:
            api_response = self.v1api.delete_namespaced_config_map(
                name,
                self.resolve_namespace(namespace),
                grace_period_seconds=0,
                propagation_policy="Background",
            )

            return api_response
        except ApiException as exc:
            # ignore error if ConfigMap is already removed
            if exc.status != 404:
                logger.error(f"failed to delete ConfigMap: {err_to_str(exc)}")
            raise exc

    def list_cfgmap(self, namespace=None, selector=""):
        try:
            resp = self.v1api.list_namespaced_config_map(
                self.resolve_namespace(namespace), watch=False, label_selector=selector
            )
        except ApiException as exc:
            logger.error(f"failed to list ConfigMaps: {err_to_str(exc)}")
            raise exc

        items = []
        for i in resp.items:
            items.append(i)
        return items

    def get_logger_pods(self, project, uid, run_kind, namespace=""):

        # As this file is imported in mlrun.runtimes, we sadly cannot have this import in the top level imports
        # as that will create an import loop.
        # TODO: Fix the import loops already!
        import mlrun.runtimes

        namespace = self.resolve_namespace(namespace)
        mpijob_crd_version = mlrun.runtimes.utils.resolve_mpijob_crd_version(
            api_context=True
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

    def create_project_service_account(self, project, service_account, namespace=""):
        namespace = self.resolve_namespace(namespace)
        k8s_service_account = client.V1ServiceAccount()
        labels = {"mlrun/project": project}
        k8s_service_account.metadata = client.V1ObjectMeta(
            name=service_account, namespace=namespace, labels=labels
        )
        try:
            api_response = self.v1api.create_namespaced_service_account(
                namespace,
                k8s_service_account,
            )
            return api_response
        except ApiException as exc:
            logger.error(f"failed to create service account: {err_to_str(exc)}")
            raise exc

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
                logger.error(f"failed to retrieve service accounts: {err_to_str(exc)}")
                raise exc
            return None

        if len(service_account.secrets) > 1:
            raise ValueError(
                f"Service account {service_account_name} has more than one secret"
            )

        return service_account.secrets[0].name

    def get_project_secret_name(self, project) -> str:
        return mlconfig.secret_stores.kubernetes.project_secret_name.format(
            project=project
        )

    def get_auth_secret_name(self, access_key: str) -> str:
        hashed_access_key = self._hash_access_key(access_key)
        return mlconfig.secret_stores.kubernetes.auth_secret_name.format(
            hashed_access_key=hashed_access_key
        )

    @staticmethod
    def _hash_access_key(access_key: str):
        return hashlib.sha224(access_key.encode()).hexdigest()

    def store_project_secrets(self, project, secrets, namespace=""):
        secret_name = self.get_project_secret_name(project)
        self.store_secrets(secret_name, secrets, namespace)

    def read_auth_secret(self, secret_name, namespace="", raise_on_not_found=False):
        namespace = self.resolve_namespace(namespace)

        try:
            secret_data = self.v1api.read_namespaced_secret(secret_name, namespace).data
        except ApiException as exc:
            logger.error(
                "Failed to read secret",
                secret_name=secret_name,
                namespace=namespace,
                exc=err_to_str(exc),
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
            mlrun.api.schemas.AuthSecretData.get_field_secret_key("username")
        )
        access_key = _get_secret_value(
            mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key")
        )

        return username, access_key

    def store_auth_secret(self, username: str, access_key: str, namespace="") -> str:
        secret_name = self.get_auth_secret_name(access_key)
        secret_data = {
            mlrun.api.schemas.AuthSecretData.get_field_secret_key("username"): username,
            mlrun.api.schemas.AuthSecretData.get_field_secret_key(
                "access_key"
            ): access_key,
        }
        self.store_secrets(
            secret_name,
            secret_data,
            namespace,
            type_=SecretTypes.v3io_fuse,
            labels={"mlrun/username": username},
        )
        return secret_name

    def store_secrets(
        self,
        secret_name,
        secrets,
        namespace="",
        type_=SecretTypes.opaque,
        labels: typing.Optional[dict] = None,
    ):
        namespace = self.resolve_namespace(namespace)
        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret doesn't exist, we'll simply create it
            if exc.status != 404:
                logger.error(f"failed to retrieve k8s secret: {err_to_str(exc)}")
                raise exc
            k8s_secret = client.V1Secret(type=type_)
            k8s_secret.metadata = client.V1ObjectMeta(
                name=secret_name, namespace=namespace, labels=labels
            )
            k8s_secret.string_data = secrets
            self.v1api.create_namespaced_secret(namespace, k8s_secret)
            return

        secret_data = k8s_secret.data.copy()
        for key, value in secrets.items():
            secret_data[key] = base64.b64encode(value.encode()).decode("utf-8")

        k8s_secret.data = secret_data
        self.v1api.replace_namespaced_secret(secret_name, namespace, k8s_secret)

    def delete_project_secrets(self, project, secrets, namespace=""):
        secret_name = self.get_project_secret_name(project)
        self.delete_secrets(secret_name, secrets, namespace)

    def delete_auth_secret(self, secret_ref: str, namespace=""):
        self.delete_secrets(secret_ref, {}, namespace)

    def delete_secrets(self, secret_name, secrets, namespace=""):
        namespace = self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret does not exist, return as if the deletion was successfully
            if exc.status == 404:
                return
            else:
                logger.error(f"failed to retrieve k8s secret: {err_to_str(exc)}")
                raise exc

        if not secrets:
            secret_data = {}
        else:
            secret_data = k8s_secret.data.copy()
            for secret in secrets:
                secret_data.pop(secret, None)

        if not secret_data:
            self.v1api.delete_namespaced_secret(secret_name, namespace)
        else:
            k8s_secret.data = secret_data
            self.v1api.replace_namespaced_secret(secret_name, namespace, k8s_secret)

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
            v3io_to_vol(name, remote, access_key, user),
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


def format_labels(labels):
    """Convert a dictionary of labels into a comma separated string"""
    if labels:
        return ",".join([f"{k}={v}" for k, v in labels.items()])
    else:
        return ""


def verify_gpu_requests_and_limits(requests_gpu: str = None, limits_gpu: str = None):
    # https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
    if requests_gpu and not limits_gpu:
        raise mlrun.errors.MLRunConflictError(
            "You cannot specify GPU requests without specifying limits"
        )
    if requests_gpu and limits_gpu and requests_gpu != limits_gpu:
        raise mlrun.errors.MLRunConflictError(
            f"When specifying both GPU requests and limits these two values must be equal, "
            f"requests_gpu={requests_gpu}, limits_gpu={limits_gpu}"
        )


def generate_preemptible_node_selector_requirements(
    node_selector_operator: str,
) -> typing.List[kubernetes.client.V1NodeSelectorRequirement]:
    """
    Generate node selector requirements based on the pre-configured node selector of the preemptible nodes.
    node selector operator represents a key's relationship to a set of values.
    Valid operators are listed in :py:class:`~mlrun.api.schemas.NodeSelectorOperator`
    :param node_selector_operator: The operator of V1NodeSelectorRequirement
    :return: List[V1NodeSelectorRequirement]
    """
    match_expressions = []
    for (
        node_selector_key,
        node_selector_value,
    ) in mlconfig.get_preemptible_node_selector().items():
        match_expressions.append(
            kubernetes.client.V1NodeSelectorRequirement(
                key=node_selector_key,
                operator=node_selector_operator,
                values=[node_selector_value],
            )
        )
    return match_expressions


def generate_preemptible_nodes_anti_affinity_terms() -> typing.List[
    kubernetes.client.V1NodeSelectorTerm
]:
    """
    Generate node selector term containing anti-affinity expressions based on the
    pre-configured node selector of the preemptible nodes.
    Use for purpose of scheduling on node only if all match_expressions are satisfied.
    This function uses a single term with potentially multiple expressions to ensure anti affinity.
    https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
    :return: List contains one nodeSelectorTerm with multiple expressions.
    """
    # import here to avoid circular imports
    from mlrun.api.schemas import NodeSelectorOperator

    # compile affinities with operator NotIn to make sure pods are not running on preemptible nodes.
    node_selector_requirements = generate_preemptible_node_selector_requirements(
        NodeSelectorOperator.node_selector_op_not_in.value
    )
    return [
        kubernetes.client.V1NodeSelectorTerm(
            match_expressions=node_selector_requirements,
        )
    ]


def generate_preemptible_nodes_affinity_terms() -> typing.List[
    kubernetes.client.V1NodeSelectorTerm
]:
    """
    Use for purpose of scheduling on node having at least one of the node selectors.
    When specifying multiple nodeSelectorTerms associated with nodeAffinity types,
    then the pod can be scheduled onto a node if at least one of the nodeSelectorTerms can be satisfied.
    :return: List of nodeSelectorTerms associated with the preemptible nodes.
    """
    # import here to avoid circular imports
    from mlrun.api.schemas import NodeSelectorOperator

    node_selector_terms = []

    # compile affinities with operator In so pods could schedule on at least one of the preemptible nodes.
    node_selector_requirements = generate_preemptible_node_selector_requirements(
        NodeSelectorOperator.node_selector_op_in.value
    )
    for expression in node_selector_requirements:
        node_selector_terms.append(
            kubernetes.client.V1NodeSelectorTerm(match_expressions=[expression])
        )
    return node_selector_terms


def generate_preemptible_tolerations() -> typing.List[kubernetes.client.V1Toleration]:
    tolerations = mlconfig.get_preemptible_tolerations()

    toleration_objects = []
    for toleration in tolerations:
        toleration_objects.append(
            kubernetes.client.V1Toleration(
                effect=toleration.get("effect", None),
                key=toleration.get("key", None),
                value=toleration.get("value", None),
                operator=toleration.get("operator", None),
                toleration_seconds=toleration.get("toleration_seconds", None)
                or toleration.get("tolerationSeconds", None),
            )
        )
    return toleration_objects
