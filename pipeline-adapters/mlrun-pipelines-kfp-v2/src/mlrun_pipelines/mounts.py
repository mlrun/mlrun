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
from typing import Optional

import mlrun
from mlrun.config import config
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.utils import logger


def v3io_cred(api="", user="", access_key=""):
    """
    Modifier function to copy local v3io env vars to container

    Usage::

        train = train_op(...)
        train.apply(use_v3io_cred())
    """
    raise NotImplementedError


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
    raise NotImplementedError


def mount_v3iod(namespace, v3io_config_configmap):
    raise NotImplementedError


def mount_pvc(pvc_name=None, volume_name="pipeline", volume_mount_path="/mnt/pipeline"):
    """
    Modifier function to apply to a PipelineTask to simplify volume, volume mount addition and
    enable better reuse of volumes, volume claims across pipelinetasks.

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

    def _mount_pvc(runtime):
        runtime.spec.update_vols_and_mounts(
            volumes=[{"PVC": {"name": pvc_name}, "name": volume_name}],
            volume_mounts=[{"mountPath": volume_mount_path, "name": volume_name}],
        )

    return _mount_pvc


def set_env_variables(env_vars_dict: Optional[dict[str, str]] = None, **kwargs):
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

    def _set_env_variables(runtime):
        for _key, _value in env_data.items():
            runtime.set_env(_key, _value)
        return runtime

    return _set_env_variables


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

    def _use_s3_cred(runtime):
        from os import environ

        _access_key = aws_access_key or environ.get(prefix + "AWS_ACCESS_KEY_ID")
        _secret_key = aws_secret_key or environ.get(prefix + "AWS_SECRET_ACCESS_KEY")
        _endpoint_url = endpoint_url or environ.get(prefix + "S3_ENDPOINT_URL")

        if _endpoint_url:
            runtime.set_env(prefix + "S3_ENDPOINT_URL", endpoint_url)
        if aws_region:
            runtime.set_env(prefix + "AWS_REGION", aws_region)
        if non_anonymous:
            runtime.set_env(prefix + "S3_NON_ANONYMOUS", "true")

        if secret_name:
            runtime.set_env_from_secret(
                prefix + "AWS_ACCESS_KEY_ID",
                secret=secret_name,
                secret_key="AWS_ACCESS_KEY_ID",
            )
            runtime.set_env_from_secret(
                prefix + "AWS_SECRET_ACCESS_KEY",
                secret=secret_name,
                secret_key="AWS_SECRET_ACCESS_KEY",
            )
        else:
            runtime.set_env(prefix + "AWS_ACCESS_KEY_ID", _access_key)
            runtime.set_env(prefix + "AWS_SECRET_ACCESS_KEY", _secret_key)
        return runtime

    return _use_s3_cred


def mount_spark_conf():
    raise NotImplementedError


def mount_secret(secret_name, mount_path, volume_name="secret", items=None):
    """Modifier function to mount kubernetes secret as files(s)

    :param secret_name:  k8s secret name
    :param mount_path:   path to mount inside the container
    :param volume_name:  unique volume name
    :param items:        unused due to lack of support on KFPv2 SDK
                         kept for backwards compatibility
    """

    def _mount_secret(runtime):
        runtime.spec.update_vols_and_mounts(
            volumes=[{"secret": {"name": secret_name}, "name": volume_name}],
            volume_mounts=[{"mountPath": mount_path, "name": volume_name}],
        )
        return runtime

    return _mount_secret


def mount_configmap(configmap_name, mount_path, volume_name="configmap", items=None):
    """Modifier function to mount kubernetes configmap as files(s)

    :param configmap_name:  k8s configmap name
    :param mount_path:      path to mount inside the container
    :param volume_name:     unique volume name
    :param items:           unused due to lack of support on KFPv2 SDK
                            kept for backwards compatibility
    """

    def _mount_configmap(runtime):
        runtime.spec.update_vols_and_mounts(
            volumes=[{"configMap": {"name": configmap_name}, "name": volume_name}],
            volume_mounts=[{"mountPath": mount_path, "name": volume_name}],
        )
        return runtime

    return _mount_configmap


def mount_hostpath(host_path, mount_path, volume_name="hostpath"):
    """Modifier function to mount kubernetes configmap as files(s)

    :param host_path:  host path
    :param mount_path:   path to mount inside the container
    :param volume_name:  unique volume name
    """
    raise NotImplementedError(
        "Support for hostPath mounting is not yet available on the KFP 2 engine"
    )


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
