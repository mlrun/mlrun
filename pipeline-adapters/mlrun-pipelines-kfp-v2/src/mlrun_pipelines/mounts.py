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

import mlrun
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
