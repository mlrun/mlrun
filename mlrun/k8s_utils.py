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
import time
from datetime import datetime
from sys import stdout

from kubernetes import client, config
from kubernetes.client.rest import ApiException

import mlrun.errors

from .config import config as mlconfig
from .platforms.iguazio import v3io_to_vol
from .utils import logger

_k8s = None


def get_k8s_helper(namespace=None, silent=False):
    """
    :param silent: set to true if you're calling this function from a code that might run from remotely (outside of a
    k8s cluster)
    """
    global _k8s
    if not _k8s:
        _k8s = K8sHelper(namespace, silent=silent)
    return _k8s


class K8sHelper:
    def __init__(self, namespace=None, config_file=None, silent=False):
        self.namespace = namespace or mlconfig.namespace
        self.config_file = config_file
        self.running_inside_kubernetes_cluster = False
        try:
            self._init_k8s_config()
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
            logger.error(f"failed to list pods: {exc}")
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
            self.del_pod(item.metadata.name, item.metadata.namespace)

    def create_pod(self, pod):
        if "pod" in dir(pod):
            pod = pod.pod
        pod.metadata.namespace = self.resolve_namespace(pod.metadata.namespace)
        try:
            resp = self.v1api.create_namespaced_pod(pod.metadata.namespace, pod)
        except ApiException as exc:
            logger.error(f"spec:\n{pod.spec}")
            logger.error(f"failed to create pod: {exc}")
            raise exc

        logger.info(f"Pod {resp.metadata.name} created")
        return resp.metadata.name, resp.metadata.namespace

    def del_pod(self, name, namespace=None):
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
                logger.error(f"failed to delete pod: {exc}")
            raise exc

    def get_pod(self, name, namespace=None, raise_on_not_found=False):
        try:
            api_response = self.v1api.read_namespaced_pod(
                name=name, namespace=self.resolve_namespace(namespace)
            )
            return api_response
        except ApiException as exc:
            if exc.status != 404:
                logger.error(f"failed to get pod: {exc}")
                raise exc
            else:
                if raise_on_not_found:
                    raise mlrun.errors.MLRunNotFoundError(f"Pod not found: {name}")
            return None

    def get_pod_status(self, name, namespace=None):
        return self.get_pod(
            name, namespace, raise_on_not_found=True
        ).status.phase.lower()

    def logs(self, name, namespace=None):
        try:
            resp = self.v1api.read_namespaced_pod_log(
                name=name, namespace=self.resolve_namespace(namespace)
            )
        except ApiException as exc:
            logger.error(f"failed to get pod logs: {exc}")
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
                logger.error(f"failed waiting for pod: {str(exc)}\n")
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
            logger.error(f"failed to create configmap: {exc}")
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
                logger.error(f"failed to delete ConfigMap: {exc}")
            raise exc

    def list_cfgmap(self, namespace=None, selector=""):
        try:
            resp = self.v1api.list_namespaced_config_map(
                self.resolve_namespace(namespace), watch=False, label_selector=selector
            )
        except ApiException as exc:
            logger.error(f"failed to list ConfigMaps: {exc}")
            raise exc

        items = []
        for i in resp.items:
            items.append(i)
        return items

    def get_logger_pods(self, project, uid, namespace=""):
        namespace = self.resolve_namespace(namespace)
        # TODO: all mlrun labels are sprinkled in a lot of places - they need to all be defined in a central,
        #  inclusive place.
        selector = f"mlrun/class,mlrun/project={project},mlrun/uid={uid}"
        pods = self.list_pods(namespace, selector=selector)
        if not pods:
            logger.error("no pod matches that uid", uid=uid)
            return

        kind = pods[0].metadata.labels.get("mlrun/class")
        results = {}
        for p in pods:
            if (
                (kind not in ["spark", "mpijob"])
                or (p.metadata.labels.get("spark-role", "") == "driver")
                # v1alpha1
                or (p.metadata.labels.get("mpi_role_type", "") == "launcher")
                # v1
                or (p.metadata.labels.get("mpi-job-role", "") == "launcher")
            ):
                results[p.metadata.name] = p.status.phase

        return results

    def create_project_service_account(self, project, service_account, namespace=""):
        namespace = self.resolve_namespace(namespace)
        k8s_service_account = client.V1ServiceAccount()
        labels = {"mlrun/project": project}
        k8s_service_account.metadata = client.V1ObjectMeta(
            name=service_account, namespace=namespace, labels=labels
        )
        try:
            api_response = self.v1api.create_namespaced_service_account(
                namespace, k8s_service_account,
            )
            return api_response
        except ApiException as exc:
            logger.error(f"failed to create service account: {exc}")
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
                logger.error(f"failed to retrieve service accounts: {exc}")
                raise exc
            return None

        if len(service_account.secrets) > 1:
            raise ValueError(
                f"Service account {service_account_name} has more than one secret"
            )

        return service_account.secrets[0].name

    def get_project_secret_name(self, project):
        return mlconfig.secret_stores.kubernetes.project_secret_name.format(
            project=project
        )

    def store_project_secrets(self, project, secrets, namespace=""):
        secret_name = self.get_project_secret_name(project)
        namespace = self.resolve_namespace(namespace)
        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret doesn't exist, we'll simply create it
            if exc.status != 404:
                logger.error(f"failed to retrieve k8s secret: {exc}")
                raise exc
            k8s_secret = client.V1Secret(type="Opaque")
            k8s_secret.metadata = client.V1ObjectMeta(
                name=secret_name, namespace=namespace
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
        namespace = self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret does not exist, return as if the deletion was successfully
            if exc.status == 404:
                return
            else:
                logger.error(f"failed to retrieve k8s secret: {exc}")
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

    def get_project_secret_keys(self, project, namespace=""):
        secret_name = self.get_project_secret_name(project)
        namespace = self.resolve_namespace(namespace)

        try:
            k8s_secret = self.v1api.read_namespaced_secret(secret_name, namespace)
        except ApiException as exc:
            # If secret doesn't exist, return empty list
            if exc.status != 404:
                logger.error(f"failed to retrieve k8s secret: {exc}")
                raise exc
            return None

        return list(k8s_secret.data.keys())


class BasePod:
    def __init__(
        self,
        task_name="",
        image=None,
        command=None,
        args=None,
        namespace="",
        kind="job",
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
        self._labels = {"mlrun/task-name": task_name, "mlrun/class": kind}
        self._annotations = {}
        self._init_container = None

    @property
    def pod(self):
        return self._get_spec()

    @property
    def init_container(self):
        return self._init_container

    @init_container.setter
    def init_container(self, container):
        self._init_container = container

    def set_init_container(
        self, image, command=None, args=None, env=None, image_pull_policy="IfNotPresent"
    ):
        if isinstance(env, dict):
            env = [client.V1EnvVar(name=k, value=v) for k, v in env.items()]
        self._init_container = client.V1Container(
            name="init",
            image=image,
            env=env,
            command=command,
            args=args,
            image_pull_policy=image_pull_policy,
        )

    def add_label(self, key, value):
        self._labels[key] = str(value)

    def add_annotation(self, key, value):
        self._annotations[key] = str(value)

    def add_volume(self, volume: client.V1Volume, mount_path, name=None):
        self._mounts.append(
            client.V1VolumeMount(name=name or volume.name, mount_path=mount_path)
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

    def mount_secret(self, name, path="/secret", items=None):
        self.add_volume(
            client.V1Volume(
                name=name,
                secret=client.V1SecretVolumeSource(secret_name=name, items=items,),
            ),
            mount_path=path,
        )

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
        )

        pod_spec = client.V1PodSpec(
            containers=[container], restart_policy="Never", volumes=self._volumes
        )

        if self._init_container:
            self._init_container.volume_mounts = self._mounts
            pod_spec.init_containers = [self._init_container]

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
    """ Convert a dictionary of labels into a comma separated string """
    if labels:
        return ",".join([f"{k}={v}" for k, v in labels.items()])
    else:
        return ""
