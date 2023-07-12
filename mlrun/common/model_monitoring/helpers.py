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
#
import mlrun.api
import mlrun.common
from mlrun.common.schemas.model_monitoring import (
    EndpointUID,
    FunctionURI,
    VersionedModel,
)
from mlrun.config import is_running_as_api


def create_model_endpoint_uid(function_uri: str, versioned_model: str):
    function_uri = FunctionURI.from_string(function_uri)
    versioned_model = VersionedModel.from_string(versioned_model)

    if (
        not function_uri.project
        or not function_uri.function
        or not versioned_model.model
    ):
        raise ValueError("Both function_uri and versioned_model have to be initialized")

    uid = EndpointUID(
        function_uri.project,
        function_uri.function,
        function_uri.tag,
        function_uri.hash_key,
        versioned_model.model,
        versioned_model.version,
    )

    return uid


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = mlrun.platforms.iguazio.parse_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def get_stream_path(project: str = None):
    # TODO: This function (as well as other methods in this file) includes both client and server side code. We will
    #  need to refactor and adjust this file in the future.
    """Get stream path from the project secret. If wasn't set, take it from the system configurations"""

    if is_running_as_api():
        # Running on API server side
        import mlrun.api.crud.secrets
        import mlrun.common.schemas

        stream_uri = mlrun.api.crud.secrets.Secrets().get_project_secret(
            project=project,
            provider=mlrun.common.schemas.secret.SecretProviderName.kubernetes,
            allow_secrets_from_k8s=True,
            secret_key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH,
        ) or mlrun.mlconf.get_model_monitoring_file_target_path(
            project=project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
            target="online",
        )

    else:
        import mlrun

        stream_uri = mlrun.get_secret_or_env(
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
        ) or mlrun.mlconf.get_model_monitoring_file_target_path(
            project=project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
            target="online",
        )

    if stream_uri.startswith("kafka://"):
        if "?topic" in stream_uri:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Custom kafka topic is not allowed"
            )
        # Add topic to stream kafka uri
        stream_uri += f"?topic=monitoring_stream_{project}"

    elif stream_uri.startswith("v3io://") and mlrun.mlconf.is_ce_mode():
        # V3IO is not supported in CE mode, generating a default http stream path
        stream_uri = mlrun.mlconf.model_endpoint_monitoring.default_http_sink

    return stream_uri


def get_connection_string(project: str = None):
    """Get endpoint store connection string from the project secret.
    If wasn't set, take it from the system configurations"""
    if is_running_as_api():
        # Running on API server side
        import mlrun.api.crud.secrets
        import mlrun.common.schemas

        return (
            mlrun.api.crud.secrets.Secrets().get_project_secret(
                project=project,
                provider=mlrun.common.schemas.secret.SecretProviderName.kubernetes,
                allow_secrets_from_k8s=True,
                secret_key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION,
            )
            or mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
        )
    else:
        # Running on stream server side
        import mlrun

        return (
            mlrun.get_secret_or_env(
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
            )
            or mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
        )
