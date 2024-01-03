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


import datetime
import typing

import mlrun
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
from mlrun.common.schemas.model_monitoring import EventFieldType
from mlrun.errors import MLRunInvalidArgumentError
from mlrun.model_monitoring.model_endpoint import ModelEndpoint
from mlrun.utils import logger

if typing.TYPE_CHECKING:
    from mlrun.db.base import RunDBInterface


def get_stream_path(project: str = None, application_name: str = None):
    """
    Get stream path from the project secret. If wasn't set, take it from the system configurations

    :param project:             Project name.
    :param application_name:    Application name, None for model_monitoring_stream.

    :return:                    Monitoring stream path to the relevant application.
    """

    stream_uri = mlrun.get_secret_or_env(
        mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
        if application_name is None
        else ""
    ) or mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
        target="online",
        application_name=application_name,
    )

    return mlrun.common.model_monitoring.helpers.parse_monitoring_stream_path(
        stream_uri=stream_uri, project=project, application_name=application_name
    )


def get_monitoring_parquet_path(
    project: str,
    kind: str = mlrun.common.schemas.model_monitoring.FileTargetKind.PARQUET,
) -> str:
    """Get model monitoring parquet target for the current project and kind. The parquet target path is based on the
    project artifact path. If project artifact path is not defined, the parquet target path will be based on MLRun
    artifact path.

    :param project:     Project name.
    :param kind:        indicate the kind of the parquet path, can be either stream_parquet or stream_controller_parquet

    :return:           Monitoring parquet target path.
    """

    project_obj = mlrun.get_or_create_project(name=project)
    artifact_path = project_obj.spec.artifact_path
    # Generate monitoring parquet path value
    parquet_path = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=kind,
        target="offline",
        artifact_path=artifact_path,
    )
    return parquet_path


def get_connection_string(secret_provider: typing.Callable = None) -> str:
    """Get endpoint store connection string from the project secret. If wasn't set, take it from the system
    configurations.

    :param secret_provider: An optional secret provider to get the connection string secret.

    :return:                Valid SQL connection string.

    """

    return (
        mlrun.get_secret_or_env(
            key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION,
            secret_provider=secret_provider,
        )
        or mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
    )


def bump_model_endpoint_last_request(
    project: str,
    model_endpoint: ModelEndpoint,
    db: "RunDBInterface",
    minutes_delta: int = 10,  # TODO: move to config - should be the same as `batch_interval`
    seconds_delta: int = 1,
) -> None:
    """
    Update the last request field of the model endpoint to be after the current last request time.

    :param project:         Project name.
    :param model_endpoint:  Model endpoint object.
    :param db:              DB interface.
    :param minutes_delta:   Minutes delta to add to the last request time.
    :param seconds_delta:   Seconds delta to add to the last request time. This is mainly to ensure that the last
                            request time is strongly greater than the previous one (with respect to the window time)
                            after adding the minutes delta.
    """
    if not model_endpoint.status.last_request:
        logger.error(
            "Model endpoint last request time is empty, cannot bump it.",
            project=project,
            endpoint_id=model_endpoint.metadata.uid,
        )
        raise MLRunInvalidArgumentError("Model endpoint last request time is empty")

    bumped_last_request = (
        datetime.datetime.fromisoformat(model_endpoint.status.last_request)
        + datetime.timedelta(
            minutes=minutes_delta,
            seconds=seconds_delta,
        )
        + datetime.timedelta(
            seconds=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
        )
    ).isoformat()
    logger.info(
        "Bumping model endpoint last request time",
        project=project,
        endpoint_id=model_endpoint.metadata.uid,
        last_request=model_endpoint.status.last_request,
        bumped_last_request=bumped_last_request,
    )
    db.patch_model_endpoint(
        project=project,
        endpoint_id=model_endpoint.metadata.uid,
        attributes={EventFieldType.LAST_REQUEST: bumped_last_request},
    )
