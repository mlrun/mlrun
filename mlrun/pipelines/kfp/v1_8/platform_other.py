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
