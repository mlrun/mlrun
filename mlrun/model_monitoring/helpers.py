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
from mlrun.common.schemas.model_monitoring import (
    EventFieldType,
    MonitoringFunctionNames,
)
from mlrun.errors import MLRunValueError
from mlrun.model_monitoring.model_endpoint import ModelEndpoint
from mlrun.utils import logger

if typing.TYPE_CHECKING:
    from mlrun.db.base import RunDBInterface
    from mlrun.projects import MlrunProject


class _BatchDict(typing.TypedDict):
    minutes: int
    hours: int
    days: int


class _MLRunNoRunsFoundError(Exception):
    pass


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
    project: "MlrunProject",
    kind: str = mlrun.common.schemas.model_monitoring.FileTargetKind.PARQUET,
) -> str:
    """Get model monitoring parquet target for the current project and kind. The parquet target path is based on the
    project artifact path. If project artifact path is not defined, the parquet target path will be based on MLRun
    artifact path.

    :param project:     Project object.
    :param kind:        indicate the kind of the parquet path, can be either stream_parquet or stream_controller_parquet

    :return:           Monitoring parquet target path.
    """
    artifact_path = project.spec.artifact_path
    # Generate monitoring parquet path value
    parquet_path = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project.name,
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


def batch_dict2timedelta(batch_dict: _BatchDict) -> datetime.timedelta:
    """
    Convert a batch dictionary to timedelta.

    :param batch_dict:  Batch dict.

    :return:            Timedelta.
    """
    return datetime.timedelta(**batch_dict)


def _get_monitoring_time_window_from_controller_run(
    project: str, db: "RunDBInterface"
) -> datetime.timedelta:
    """
    Get timedelta for the controller to run.

    :param project: Project name.
    :param db:      DB interface.

    :return:    Timedelta for the controller to run.
    """
    run_name = MonitoringFunctionNames.APPLICATION_CONTROLLER
    runs = db.list_runs(project=project, name=run_name, sort=True)
    if not runs:
        raise _MLRunNoRunsFoundError(f"No {run_name} runs were found")
    last_run = runs[0]
    try:
        batch_dict = last_run["spec"]["parameters"]["batch_intervals_dict"]
    except KeyError:
        raise MLRunValueError(
            f"Could not find `batch_intervals_dict` in {run_name} run"
        )
    return batch_dict2timedelta(batch_dict)


def update_model_endpoint_last_request(
    project: str,
    model_endpoint: ModelEndpoint,
    current_request: datetime,
    db: "RunDBInterface",
) -> None:
    """
    Update the last request field of the model endpoint to be after the current request time.

    :param project:         Project name.
    :param model_endpoint:  Model endpoint object.
    :param current_request: current request time
    :param db:              DB interface.
    """
    if model_endpoint.spec.stream_path != "":
        current_request = current_request.isoformat()
        logger.info(
            "Update model endpoint last request time (EP with serving)",
            project=project,
            endpoint_id=model_endpoint.metadata.uid,
            last_request=model_endpoint.status.last_request,
            current_request=current_request,
        )
        db.patch_model_endpoint(
            project=project,
            endpoint_id=model_endpoint.metadata.uid,
            attributes={EventFieldType.LAST_REQUEST: current_request},
        )
    else:
        try:
            time_window = _get_monitoring_time_window_from_controller_run(project, db)
        except _MLRunNoRunsFoundError:
            logger.debug(
                "Not bumping model endpoint last request time - no controller runs were found"
            )
            return

        bumped_last_request = (
            current_request
            + time_window
            + datetime.timedelta(
                seconds=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
            )
        ).isoformat()
        logger.info(
            "Bumping model endpoint last request time (EP without serving)",
            project=project,
            endpoint_id=model_endpoint.metadata.uid,
            last_request=model_endpoint.status.last_request,
            current_request=current_request.isoformat(),
            bumped_last_request=bumped_last_request,
        )
        db.patch_model_endpoint(
            project=project,
            endpoint_id=model_endpoint.metadata.uid,
            attributes={EventFieldType.LAST_REQUEST: bumped_last_request},
        )
