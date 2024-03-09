# Copyright 2024 Iguazio
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
import os

import kfp.dsl
import semver
from mlrun_pipelines.common.mounts import _enrich_and_validate_v3io_mounts

from mlrun.config import config
from mlrun.config import config as mlconf
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.platforms.iguazio import v3io_to_vol
from mlrun.utils import logger


def v3io_cred(api="", user="", access_key=""):
    """
    Modifier function to copy local v3io env vars to container

    Usage::

        train = train_op(...)
        train.apply(use_v3io_cred())
    """

    # TODO: this is a guy that uses modifiers for mounting v3io
    def _use_v3io_cred(container_op: kfp.dsl.ContainerOp):
        from os import environ

        from kubernetes import client as k8s_client

        web_api = api or environ.get("V3IO_API") or mlconf.v3io_api
        _user = user or environ.get("V3IO_USERNAME")
        _access_key = access_key or environ.get("V3IO_ACCESS_KEY")
        v3io_framesd = mlconf.v3io_framesd or environ.get("V3IO_FRAMESD")

        return (
            container_op.container.add_env_variable(
                k8s_client.V1EnvVar(name="V3IO_API", value=web_api)
            )
            .add_env_variable(k8s_client.V1EnvVar(name="V3IO_USERNAME", value=_user))
            .add_env_variable(
                k8s_client.V1EnvVar(name="V3IO_ACCESS_KEY", value=_access_key)
            )
            .add_env_variable(
                k8s_client.V1EnvVar(name="V3IO_FRAMESD", value=v3io_framesd)
            )
        )

    return _use_v3io_cred


def mount_v3io(
    name="v3io",
    remote="",
    access_key="",
    user="",
    secret=None,
    volume_mounts=None,
):
    """Modifier function to apply to a Container Op to volume mount a v3io path

    :param name:            the volume name
    :param remote:          the v3io path to use for the volume. ~/ prefix will be replaced with /users/<username>/
    :param access_key:      the access key used to auth against v3io. if not given V3IO_ACCESS_KEY env var will be used
    :param user:            the username used to auth against v3io. if not given V3IO_USERNAME env var will be used
    :param secret:          k8s secret name which would be used to get the username and access key to auth against v3io.
    :param volume_mounts:   list of VolumeMount. empty volume mounts & remote will default to mount /v3io & /User.
    """
    volume_mounts, user = _enrich_and_validate_v3io_mounts(
        remote=remote,
        volume_mounts=volume_mounts,
        user=user,
    )

    def _attach_volume_mounts_and_creds(container_op: kfp.dsl.ContainerOp):
        from kubernetes import client as k8s_client

        vol = v3io_to_vol(name, remote, access_key, user, secret=secret)
        container_op.add_volume(vol)
        for volume_mount in volume_mounts:
            container_op.container.add_volume_mount(
                k8s_client.V1VolumeMount(
                    mount_path=volume_mount.path,
                    sub_path=volume_mount.sub_path,
                    name=name,
                )
            )

        if not secret:
            container_op = v3io_cred(access_key=access_key, user=user)(container_op)
        return container_op

    return _attach_volume_mounts_and_creds


def mount_spark_conf():
    def _mount_spark(container_op: kfp.dsl.ContainerOp):
        from kubernetes import client as k8s_client

        container_op.container.add_volume_mount(
            k8s_client.V1VolumeMount(
                name="spark-master-config", mount_path="/etc/config/spark"
            )
        )
        return container_op

    return _mount_spark


def mount_v3iod(namespace, v3io_config_configmap):
    def _mount_v3iod(container_op: kfp.dsl.ContainerOp):
        from kubernetes import client as k8s_client

        def add_vol(name, mount_path, host_path):
            vol = k8s_client.V1Volume(
                name=name,
                host_path=k8s_client.V1HostPathVolumeSource(path=host_path, type=""),
            )
            container_op.add_volume(vol)
            container_op.container.add_volume_mount(
                k8s_client.V1VolumeMount(mount_path=mount_path, name=name)
            )

        # this is a legacy path for the daemon shared memory
        host_path = "/dev/shm/"

        # path to shared memory for daemon was changed in Iguazio 3.2.3-b1
        igz_version = mlconf.get_parsed_igz_version()
        if igz_version and igz_version >= semver.VersionInfo.parse("3.2.3-b1"):
            host_path = "/var/run/iguazio/dayman-shm/"
        add_vol(name="shm", mount_path="/dev/shm", host_path=host_path + namespace)

        add_vol(
            name="v3iod-comm",
            mount_path="/var/run/iguazio/dayman",
            host_path="/var/run/iguazio/dayman/" + namespace,
        )

        vol = k8s_client.V1Volume(
            name="daemon-health", empty_dir=k8s_client.V1EmptyDirVolumeSource()
        )
        container_op.add_volume(vol)
        container_op.container.add_volume_mount(
            k8s_client.V1VolumeMount(
                mount_path="/var/run/iguazio/daemon_health", name="daemon-health"
            )
        )

        vol = k8s_client.V1Volume(
            name="v3io-config",
            config_map=k8s_client.V1ConfigMapVolumeSource(
                name=v3io_config_configmap, default_mode=420
            ),
        )
        container_op.add_volume(vol)
        container_op.container.add_volume_mount(
            k8s_client.V1VolumeMount(mount_path="/etc/config/v3io", name="v3io-config")
        )

        container_op.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="CURRENT_NODE_IP",
                value_from=k8s_client.V1EnvVarSource(
                    field_ref=k8s_client.V1ObjectFieldSelector(
                        api_version="v1", field_path="status.hostIP"
                    )
                ),
            )
        )
        container_op.container.add_env_variable(
            k8s_client.V1EnvVar(
                name="IGZ_DATA_CONFIG_FILE", value="/igz/java/conf/v3io.conf"
            )
        )

        return container_op

    return _mount_v3iod


def mount_s3(
    secret_name=None,
    aws_access_key="",
    aws_secret_key="",
    endpoint_url=None,
    prefix="",
    aws_region=None,
    non_anonymous=False,
):
    """Modifier function to add s3 env vars or secrets to container

    **Warning:**
    Using this function to configure AWS credentials will expose these credentials in the pod spec of the runtime
    created. It is recommended to use the `secret_name` parameter, or set the credentials as project-secrets and avoid
    using this function.

    :param secret_name: kubernetes secret name (storing the access/secret keys)
    :param aws_access_key: AWS_ACCESS_KEY_ID value. If this parameter is not specified and AWS_ACCESS_KEY_ID env.
                            variable is defined, the value will be taken from the env. variable
    :param aws_secret_key: AWS_SECRET_ACCESS_KEY value. If this parameter is not specified and AWS_SECRET_ACCESS_KEY
                            env. variable is defined, the value will be taken from the env. variable
    :param endpoint_url: s3 endpoint address (for non AWS s3)
    :param prefix: string prefix to add before the env var name (for working with multiple s3 data stores)
    :param aws_region: amazon region
    :param non_anonymous: force the S3 API to use non-anonymous connection, even if no credentials are provided
        (for authenticating externally, such as through IAM instance-roles)
    """

    if secret_name and (aws_access_key or aws_secret_key):
        raise MLRunInvalidArgumentError(
            "can use k8s_secret for credentials or specify them (aws_access_key, aws_secret_key) not both"
        )

    if not secret_name and (
        aws_access_key
        or os.environ.get(prefix + "AWS_ACCESS_KEY_ID")
        or aws_secret_key
        or os.environ.get(prefix + "AWS_SECRET_ACCESS_KEY")
    ):
        logger.warning(
            "it is recommended to use k8s secret (specify secret_name), "
            "specifying the aws_access_key/aws_secret_key directly is unsafe"
        )

    def _use_s3_cred(container_op):
        from os import environ

        from kubernetes import client as k8s_client

        _access_key = aws_access_key or environ.get(prefix + "AWS_ACCESS_KEY_ID")
        _secret_key = aws_secret_key or environ.get(prefix + "AWS_SECRET_ACCESS_KEY")
        _endpoint_url = endpoint_url or environ.get(prefix + "S3_ENDPOINT_URL")

        container = container_op.container
        if _endpoint_url:
            container.add_env_variable(
                k8s_client.V1EnvVar(name=prefix + "S3_ENDPOINT_URL", value=endpoint_url)
            )
        if aws_region:
            container.add_env_variable(
                k8s_client.V1EnvVar(name=prefix + "AWS_REGION", value=aws_region)
            )
        if non_anonymous:
            container.add_env_variable(
                k8s_client.V1EnvVar(name=prefix + "S3_NON_ANONYMOUS", value="true")
            )

        if secret_name:
            container.add_env_variable(
                k8s_client.V1EnvVar(
                    name=prefix + "AWS_ACCESS_KEY_ID",
                    value_from=k8s_client.V1EnvVarSource(
                        secret_key_ref=k8s_client.V1SecretKeySelector(
                            name=secret_name, key="AWS_ACCESS_KEY_ID"
                        )
                    ),
                )
            ).add_env_variable(
                k8s_client.V1EnvVar(
                    name=prefix + "AWS_SECRET_ACCESS_KEY",
                    value_from=k8s_client.V1EnvVarSource(
                        secret_key_ref=k8s_client.V1SecretKeySelector(
                            name=secret_name, key="AWS_SECRET_ACCESS_KEY"
                        )
                    ),
                )
            )

        else:
            return container_op.add_env_variable(
                k8s_client.V1EnvVar(
                    name=prefix + "AWS_ACCESS_KEY_ID", value=_access_key
                )
            ).add_env_variable(
                k8s_client.V1EnvVar(
                    name=prefix + "AWS_SECRET_ACCESS_KEY", value=_secret_key
                )
            )

    return _use_s3_cred


def mount_pvc(pvc_name=None, volume_name="pipeline", volume_mount_path="/mnt/pipeline"):
    """
    Modifier function to apply to a Container Op to simplify volume, volume mount addition and
    enable better reuse of volumes, volume claims across container ops.

    Usage::

        train = train_op(...)
        train.apply(mount_pvc('claim-name', 'pipeline', '/mnt/pipeline'))
    """
    if "MLRUN_PVC_MOUNT" in os.environ:
        mount = os.environ.get("MLRUN_PVC_MOUNT")
        items = mount.split(":")
        if len(items) != 2:
            raise MLRunInvalidArgumentError(
                "MLRUN_PVC_MOUNT should include <pvc-name>:<mount-path>"
            )
        pvc_name = items[0]
        volume_mount_path = items[1]

    if not pvc_name:
        raise MLRunInvalidArgumentError(
            "No PVC name: use the pvc_name parameter or configure the MLRUN_PVC_MOUNT environment variable"
        )

    def _mount_pvc(task):
        from kubernetes import client as k8s_client

        local_pvc = k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)
        return task.add_volume(
            k8s_client.V1Volume(name=volume_name, persistent_volume_claim=local_pvc)
        ).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path=volume_mount_path, name=volume_name)
        )

    return _mount_pvc


def auto_mount(pvc_name="", volume_mount_path="", volume_name=None):
    """choose the mount based on env variables and params

    volume will be selected by the following order:
    - k8s PVC volume when both pvc_name and volume_mount_path are set
    - k8s PVC volume when env var is set: MLRUN_PVC_MOUNT=<pvc-name>:<mount-path>
    - k8s PVC volume if it's configured as the auto mount type
    - iguazio v3io volume when V3IO_ACCESS_KEY and V3IO_USERNAME env vars are set
    """
    if pvc_name and volume_mount_path:
        return mount_pvc(
            pvc_name=pvc_name,
            volume_mount_path=volume_mount_path,
            volume_name=volume_name or "shared-persistency",
        )
    if "MLRUN_PVC_MOUNT" in os.environ:
        return mount_pvc(
            volume_name=volume_name or "shared-persistency",
        )
    # In the case of MLRun-kit when working remotely, no env variables will be defined but auto-mount
    # parameters may still be declared - use them in that case.
    if config.storage.auto_mount_type == "pvc":
        return mount_pvc(**config.get_storage_auto_mount_params())
    if "V3IO_ACCESS_KEY" in os.environ:
        return mount_v3io(name=volume_name or "v3io")

    raise ValueError("failed to auto mount, need to set env vars")


def mount_secret(secret_name, mount_path, volume_name="secret", items=None):
    """Modifier function to mount kubernetes secret as files(s)

    :param secret_name:  k8s secret name
    :param mount_path:   path to mount inside the container
    :param volume_name:  unique volume name
    :param items:        If unspecified, each key-value pair in the Data field
                         of the referenced Secret will be projected into the
                         volume as a file whose name is the key and content is
                         the value.
                         If specified, the listed keys will be projected into
                         the specified paths, and unlisted keys will not be
                         present.
    """

    def _mount_secret(task):
        from kubernetes import client as k8s_client

        vol = k8s_client.V1SecretVolumeSource(secret_name=secret_name, items=items)
        return task.add_volume(
            k8s_client.V1Volume(name=volume_name, secret=vol)
        ).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path=mount_path, name=volume_name)
        )

    return _mount_secret


def mount_configmap(configmap_name, mount_path, volume_name="configmap", items=None):
    """Modifier function to mount kubernetes configmap as files(s)

    :param configmap_name:  k8s configmap name
    :param mount_path:      path to mount inside the container
    :param volume_name:     unique volume name
    :param items:           If unspecified, each key-value pair in the Data field
                            of the referenced Configmap will be projected into the
                            volume as a file whose name is the key and content is
                            the value.
                            If specified, the listed keys will be projected into
                            the specified paths, and unlisted keys will not be
                            present.
    """

    def _mount_configmap(task):
        from kubernetes import client as k8s_client

        vol = k8s_client.V1ConfigMapVolumeSource(name=configmap_name, items=items)
        return task.add_volume(
            k8s_client.V1Volume(name=volume_name, config_map=vol)
        ).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path=mount_path, name=volume_name)
        )

    return _mount_configmap


def mount_hostpath(host_path, mount_path, volume_name="hostpath"):
    """Modifier function to mount kubernetes configmap as files(s)

    :param host_path:  host path
    :param mount_path:   path to mount inside the container
    :param volume_name:  unique volume name
    """

    def _mount_hostpath(task):
        from kubernetes import client as k8s_client

        return task.add_volume(
            k8s_client.V1Volume(
                name=volume_name,
                host_path=k8s_client.V1HostPathVolumeSource(path=host_path, type=""),
            )
        ).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path=mount_path, name=volume_name)
        )

    return _mount_hostpath


def set_env_variables(env_vars_dict: dict[str, str] = None, **kwargs):
    """
    Modifier function to apply a set of environment variables to a runtime. Variables may be passed
    as either a dictionary of name-value pairs, or as arguments to the function.
    See `KubeResource.apply` for more information on modifiers.

    Usage::

        function.apply(set_env_variables({"ENV1": "value1", "ENV2": "value2"}))
        or
        function.apply(set_env_variables(ENV1=value1, ENV2=value2))

    :param env_vars_dict: dictionary of env. variables
    :param kwargs: environment variables passed as args
    """

    env_data = env_vars_dict.copy() if env_vars_dict else {}
    for key, value in kwargs.items():
        env_data[key] = value

    def _set_env_variables(container_op: kfp.dsl.ContainerOp):
        from kubernetes import client as k8s_client

        for _key, _value in env_data.items():
            container_op.container.add_env_variable(
                k8s_client.V1EnvVar(name=_key, value=_value)
            )
        return container_op

    return _set_env_variables
