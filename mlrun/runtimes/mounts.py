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

import os
import typing
from collections import namedtuple

from mlrun.config import config
from mlrun.config import config as mlconf
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.platforms.iguazio import v3io_to_vol
from mlrun.utils import logger

if typing.TYPE_CHECKING:
    from mlrun.runtimes import KubeResource


VolumeMount = namedtuple("Mount", ["path", "sub_path"])


def v3io_cred(
    api: str = "",
    user: str = "",
    access_key: str = "",
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """
    Modifier function to copy local v3io env vars to container

    Usage::

        train = train_op(...)
        train.apply(use_v3io_cred())
    """

    def _use_v3io_cred(runtime: "KubeResource"):
        web_api = api or os.environ.get("V3IO_API") or mlconf.v3io_api
        _user = user or os.environ.get("V3IO_USERNAME")
        _access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        v3io_framesd = mlconf.v3io_framesd or os.environ.get("V3IO_FRAMESD")

        runtime.set_envs(
            {
                "V3IO_API": web_api,
                "V3IO_USERNAME": _user,
                "V3IO_ACCESS_KEY": _access_key,
                "V3IO_FRAMESD": v3io_framesd,
            },
        )

        return runtime

    return _use_v3io_cred


def mount_v3io(
    name: str = "v3io",
    remote: str = "",
    access_key: str = "",
    user: str = "",
    secret: typing.Optional[str] = None,
    volume_mounts: typing.Optional[list[VolumeMount]] = None,
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """Modifier function to apply to a Container Op to volume mount a v3io path

    :param name: the volume name
    :param remote: the v3io path to use for the volume (~/ prefix will be replaced with /users/<username>/)
    :param access_key: the access key used to auth against v3io (default: V3IO_ACCESS_KEY env var)
    :param user: the username used to auth against v3io (default: V3IO_USERNAME env var)
    :param secret: k8s secret name for the username and access key
    :param volume_mounts: list of VolumeMount; if empty, defaults to mounting /v3io and /User
    """
    volume_mounts, user = _enrich_and_validate_v3io_mounts(
        remote=remote,
        volume_mounts=volume_mounts,
        user=user,
    )

    def _attach_volume_mounts_and_creds(runtime: "KubeResource"):
        vol = v3io_to_vol(name, remote, access_key, user, secret=secret)
        runtime.spec.with_volumes(vol)

        for volume_mount in volume_mounts:
            runtime.spec.with_volume_mounts(
                {
                    "mountPath": volume_mount.path,
                    "name": name,
                    "subPath": volume_mount.sub_path,
                }
            )

        if not secret:
            runtime = v3io_cred(access_key=access_key, user=user)(runtime)
        return runtime

    return _attach_volume_mounts_and_creds


def mount_spark_conf() -> typing.Callable[["KubeResource"], "KubeResource"]:
    """Modifier function to mount Spark configuration."""

    def _mount_spark(runtime: "KubeResource"):
        runtime.spec.with_volume_mounts(
            {
                "mountPath": "/etc/config/spark",
                "name": "spark-master-config",
            }
        )
        return runtime

    return _mount_spark


def mount_v3iod(
    namespace: str, v3io_config_configmap: str
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """Modifier function to mount v3iod configuration."""

    def _mount_v3iod(runtime: "KubeResource"):
        def add_vol(name, mount_path, host_path):
            runtime.spec.with_volumes(
                {
                    "name": name,
                    "hostPath": {
                        "path": host_path,
                        "type": "",
                    },
                }
            )
            runtime.spec.with_volume_mounts(
                {
                    "mountPath": mount_path,
                    "name": name,
                }
            )

        add_vol(
            name="shm",
            mount_path="/dev/shm",
            host_path=f"/var/run/iguazio/dayman-shm/{namespace}",
        )
        add_vol(
            name="v3iod-comm",
            mount_path="/var/run/iguazio/dayman",
            host_path="/var/run/iguazio/dayman/" + namespace,
        )

        # Add daemon-health and v3io-config volumes
        runtime.spec.with_volumes(
            [
                {
                    "name": "daemon-health",
                    "emptyDir": {},
                },
                {
                    "name": "v3io-config",
                    "configMap": {
                        "name": v3io_config_configmap,
                        "defaultMode": 420,
                    },
                },
            ]
        )

        # Add volume mounts
        runtime.spec.with_volume_mounts(
            [
                {
                    "mountPath": "/var/run/iguazio/daemon_health",
                    "name": "daemon-health",
                },
                {
                    "mountPath": "/etc/config/v3io",
                    "name": "v3io-config",
                },
            ]
        )

        # Add environment variables
        runtime.set_envs(
            {
                "CURRENT_NODE_IP": {
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": "status.hostIP",
                        }
                    },
                },
                "IGZ_DATA_CONFIG_FILE": "/igz/java/conf/v3io.conf",
            }
        )

        return runtime

    return _mount_v3iod


def mount_s3(
    secret_name: typing.Optional[str] = None,
    aws_access_key: str = "",
    aws_secret_key: str = "",
    endpoint_url: typing.Optional[str] = None,
    prefix: str = "",
    aws_region: typing.Optional[str] = None,
    non_anonymous: bool = False,
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """Modifier function to add s3 env vars or secrets to container

    :param secret_name: Kubernetes secret name for credentials
    :param aws_access_key: AWS_ACCESS_KEY_ID value (default: env variable)
    :param aws_secret_key: AWS_SECRET_ACCESS_KEY value (default: env variable)
    :param endpoint_url: s3 endpoint address (for non-AWS s3)
    :param prefix: prefix to add before the env var name (for multiple s3 data stores)
    :param aws_region: Amazon region
    :param non_anonymous: use non-anonymous connection even if no credentials are provided
            (for authenticating externally, such as through IAM instance-roles)

    """

    if secret_name and (aws_access_key or aws_secret_key):
        raise MLRunInvalidArgumentError(
            "Can use k8s_secret for credentials or specify them (aws_access_key, aws_secret_key) not both."
        )

    if not secret_name and (
        aws_access_key
        or os.environ.get(prefix + "AWS_ACCESS_KEY_ID")
        or aws_secret_key
        or os.environ.get(prefix + "AWS_SECRET_ACCESS_KEY")
    ):
        logger.warning(
            "It is recommended to use k8s secret (specify secret_name), "
            "specifying aws_access_key/aws_secret_key directly is unsafe."
        )

    def _use_s3_cred(runtime: "KubeResource"):
        _access_key = aws_access_key or os.environ.get(prefix + "AWS_ACCESS_KEY_ID")
        _secret_key = aws_secret_key or os.environ.get(prefix + "AWS_SECRET_ACCESS_KEY")
        _endpoint_url = endpoint_url or os.environ.get(prefix + "S3_ENDPOINT_URL")

        if _endpoint_url:
            runtime.set_env(prefix + "S3_ENDPOINT_URL", _endpoint_url)
        if aws_region:
            runtime.set_env(prefix + "AWS_REGION", aws_region)
        if non_anonymous:
            runtime.set_env(prefix + "S3_NON_ANONYMOUS", "true")

        if secret_name:
            runtime.set_envs(
                {
                    f"{prefix}AWS_ACCESS_KEY_ID": {
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": secret_name,
                                "key": "AWS_ACCESS_KEY_ID",
                            }
                        }
                    },
                    f"{prefix}AWS_SECRET_ACCESS_KEY": {
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": secret_name,
                                "key": "AWS_SECRET_ACCESS_KEY",
                            }
                        },
                    },
                }
            )
        else:
            runtime.set_envs(
                {
                    f"{prefix}AWS_ACCESS_KEY_ID": _access_key,
                    f"{prefix}AWS_SECRET_ACCESS_KEY": _secret_key,
                },
            )

        return runtime

    return _use_s3_cred


def mount_pvc(
    pvc_name: typing.Optional[str] = None,
    volume_name: str = "pipeline",
    volume_mount_path: str = "/mnt/pipeline",
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """
    Modifier function to mount a PVC volume in the container, simplifying volume and volume mount addition.

    Usage::

        train = train_op(...)
        train.apply(mount_pvc("claim-name", "pipeline", "/mnt/pipeline"))
    """
    if not pvc_name:
        # Try to get the PVC mount configuration from the environment variable
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
        # The PVC name is still not set, raise an error
        raise MLRunInvalidArgumentError(
            "No PVC name: use the pvc_name parameter or configure the MLRUN_PVC_MOUNT environment variable"
        )

    def _mount_pvc(runtime: "KubeResource"):
        local_pvc = {"claimName": pvc_name}

        runtime.spec.with_volumes(
            [
                {
                    "name": volume_name,
                    "persistentVolumeClaim": local_pvc,
                }
            ]
        )
        runtime.spec.with_volume_mounts(
            {
                "mountPath": volume_mount_path,
                "name": volume_name,
            }
        )

        return runtime

    return _mount_pvc


def auto_mount(
    pvc_name: str = "",
    volume_mount_path: str = "",
    volume_name: typing.Optional[str] = None,
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """Choose the mount based on env variables and params

    Volume will be selected by the following order:

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
    # In the case of CE when working remotely, no env variables will be defined but auto-mount
    # parameters may still be declared - use them in that case.
    if config.storage.auto_mount_type == "pvc":
        return mount_pvc(**config.get_storage_auto_mount_params())
    if "V3IO_ACCESS_KEY" in os.environ:
        return mount_v3io(name=volume_name or "v3io")

    raise ValueError("Failed to auto mount, need to set env vars")


def mount_secret(
    secret_name: str,
    mount_path: str,
    volume_name: str = "secret",
    items: typing.Optional[list[dict]] = None,
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """
    Modifier function to mount a Kubernetes secret as file(s).

    :param secret_name: Kubernetes secret name
    :param mount_path: Path inside the container to mount
    :param volume_name: Unique volume name
    :param items:        If unspecified, each key-value pair in the Data field
                         of the referenced Secret will be projected into the
                         volume as a file whose name is the key and content is
                         the value.
                         If specified, the listed keys will be projected into
                         the specified paths, and unlisted keys will not be
                         present."""

    def _mount_secret(runtime: "KubeResource"):
        # Define the secret volume source
        secret_volume_source = {
            "secretName": secret_name,
            "items": items,
        }

        # Add the secret volume
        runtime.spec.with_volumes(
            {
                "name": volume_name,
                "secret": secret_volume_source,
            }
        )

        # Add the volume mount
        runtime.spec.with_volume_mounts(
            {
                "mountPath": mount_path,
                "name": volume_name,
            }
        )

        return runtime

    return _mount_secret


def mount_configmap(
    configmap_name: str,
    mount_path: str,
    volume_name: str = "configmap",
    items: typing.Optional[list[dict]] = None,
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """
    Modifier function to mount a Kubernetes ConfigMap as file(s).

    :param configmap_name: Kubernetes ConfigMap name
    :param mount_path: Path inside the container to mount
    :param volume_name: Unique volume name
    :param items:           If unspecified, each key-value pair in the Data field
                            of the referenced Configmap will be projected into the
                            volume as a file whose name is the key and content is
                            the value.
                            If specified, the listed keys will be projected into
                            the specified paths, and unlisted keys will not be
                            present."""

    def _mount_configmap(runtime: "KubeResource"):
        # Construct the configMap dictionary
        config_map_dict = {
            "name": configmap_name,
        }
        if items is not None:
            config_map_dict["items"] = items

        vol = {
            "name": volume_name,
            "configMap": config_map_dict,
        }

        runtime.spec.with_volumes(vol)
        runtime.spec.with_volume_mounts(
            {
                "mountPath": mount_path,
                "name": volume_name,
            }
        )

        return runtime

    return _mount_configmap


def mount_hostpath(
    host_path: str,
    mount_path: str,
    volume_name: str = "hostpath",
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """
    Modifier function to mount a host path inside a Kubernetes container.

    :param host_path: Host path on the node to be mounted.
    :param mount_path: Path inside the container where the volume will be mounted.
    :param volume_name: Unique name for the volume.
    """

    def _mount_hostpath(runtime: "KubeResource") -> "KubeResource":
        runtime.spec.with_volumes(
            {
                "name": volume_name,
                "hostPath": {
                    "path": host_path,
                    "type": "",
                },
            }
        )
        runtime.spec.with_volume_mounts(
            {
                "mountPath": mount_path,
                "name": volume_name,
            }
        )

        return runtime

    return _mount_hostpath


def set_env_variables(
    env_vars_dict: typing.Optional[dict[str, str]] = None, **kwargs
) -> typing.Callable[["KubeResource"], "KubeResource"]:
    """
    Modifier function to apply a set of environment variables to a runtime. Variables may be passed
    as either a dictionary of name-value pairs, or as arguments to the function.
    See `KubeResource.apply` for more information on modifiers.

    Usage::

        function.apply(set_env_variables({"ENV1": "value1", "ENV2": "value2"}))
        or
        function.apply(set_env_variables(ENV1=value1, ENV2=value2))

    :param env_vars_dict: dictionary of environment variables
    :param kwargs: environment variables passed as arguments
    """

    env_data = env_vars_dict.copy() if env_vars_dict else {}
    for key, value in kwargs.items():
        env_data[key] = value

    def _set_env_variables(runtime: "KubeResource"):
        runtime.set_envs(env_data)

        return runtime

    return _set_env_variables


def _enrich_and_validate_v3io_mounts(
    remote: str = "",
    volume_mounts: typing.Optional[list[VolumeMount]] = None,
    user: str = "",
) -> tuple[list[VolumeMount], str]:
    if volume_mounts is None:
        volume_mounts = []
    if remote and not volume_mounts:
        raise MLRunInvalidArgumentError(
            "volume_mounts must be specified when remote is given"
        )

    # Empty remote & volume_mounts defaults are volume mounts of /v3io and /User
    if not remote and not volume_mounts:
        user = _resolve_mount_user(user)
        if not user:
            raise MLRunInvalidArgumentError(
                "user name/env must be specified when using empty remote and volume_mounts"
            )
        volume_mounts = [
            VolumeMount(path="/v3io", sub_path=""),
            VolumeMount(path="/User", sub_path="users/" + user),
        ]

    if not isinstance(volume_mounts, list) and any(
        [not isinstance(x, VolumeMount) for x in volume_mounts]
    ):
        raise TypeError("mounts should be a list of Mount")

    return volume_mounts, user


def _resolve_mount_user(user: typing.Optional[str] = None):
    return user or os.environ.get("V3IO_USERNAME")
