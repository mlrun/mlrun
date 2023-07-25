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


import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
from mlrun.config import is_running_as_api


def get_stream_path(project: str = None):
    """Get stream path from the project secret. If wasn't set, take it from the system configurations"""

    stream_uri = mlrun.get_secret_or_env(
        mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
    ) or mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
        target="online",
    )

    return mlrun.common.model_monitoring.helpers.parse_monitoring_stream_path(
        stream_uri=stream_uri, project=project
    )


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
