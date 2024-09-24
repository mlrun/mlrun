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

import json
import typing

import sqlalchemy.orm

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
import mlrun.model_monitoring
import mlrun.model_monitoring.db.stores
import server.api.crud.secrets


def json_loads_if_not_none(field: typing.Any) -> typing.Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )


def get_access_key(auth_info: mlrun.common.schemas.AuthInfo):
    """
    Get access key from the current data session. This method is usually used to verify that the session
    is valid and contains an access key.

    param auth_info: The auth info of the request.

    :return: Access key as a string.
    """
    access_key = auth_info.data_session
    if not access_key:
        raise mlrun.errors.MLRunBadRequestError("Data session is missing")
    return access_key


def get_monitoring_parquet_path(
    db_session: sqlalchemy.orm.Session,
    project: str,
    kind: str = mlrun.common.schemas.model_monitoring.FileTargetKind.PARQUET,
) -> str:
    """Get model monitoring parquet target for the current project. The parquet target path is based on the
    project artifact path. If project artifact path is not defined, the parquet target path will be based on MLRun
    artifact path.

    :param db_session: A session that manages the current dialog with the database. Will be used in this function
                       to get the project record from DB.
    :param project:    Project name.
    :param kind:       indicate the kind of the parquet path, can be either stream_parquet or stream_controller_parquet

    :return:           Monitoring parquet target path.
    """

    # Get the artifact path from the project record that was stored in the DB
    project_obj = server.api.crud.projects.Projects().get_project(
        session=db_session, name=project
    )
    artifact_path = project_obj.spec.artifact_path
    # Generate monitoring parquet path value
    parquet_path = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=kind,
        target="offline",
        artifact_path=artifact_path,
    )
    return parquet_path


def get_stream_path(
    project: str,
    function_name: str = mm_constants.MonitoringFunctionNames.STREAM,
    stream_uri: typing.Optional[str] = None,
) -> str:
    """
    Get stream path from the project secret. If wasn't set, take it from the system configurations.

    :param project:       Project name.
    :param function_name: Application name. Default is model_monitoring_stream.
    :param stream_uri:    Stream URI. If not provided, it will be taken from the project secret.

    :return:              Monitoring stream path to the relevant application.
    """

    stream_uri = stream_uri or server.api.crud.secrets.Secrets().get_project_secret(
        project=project,
        provider=mlrun.common.schemas.secret.SecretProviderName.kubernetes,
        allow_secrets_from_k8s=True,
        secret_key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH,
    )

    if not stream_uri or stream_uri == "v3io":
        stream_uri = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
            target="online",
            function_name=function_name,
        )

    return mlrun.common.model_monitoring.helpers.parse_monitoring_stream_path(
        stream_uri=stream_uri, project=project, function_name=function_name
    )


def get_store_object(project: str) -> mlrun.model_monitoring.db.stores.StoreBase:
    """Handle the get store object function for the server side, using the project secret provider."""
    return mlrun.model_monitoring.get_store_object(
        project=project,
        secret_provider=server.api.crud.secrets.get_project_secret_provider(
            project=project
        ),
    )
