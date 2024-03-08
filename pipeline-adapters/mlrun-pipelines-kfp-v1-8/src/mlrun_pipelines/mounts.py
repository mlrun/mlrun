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

import mlrun
from mlrun.config import config as mlconf
from mlrun.platforms.iguazio import _enrich_and_validate_v3io_mounts, v3io_to_vol
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
        raise mlrun.errors.MLRunInvalidArgumentError(
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
