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

import numpy as np
import pandas as pd

import mlrun
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
from mlrun.common.schemas.model_monitoring import (
    EventFieldType,
)
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
    _compose_full_name,
)
from mlrun.model_monitoring.model_endpoint import ModelEndpoint
from mlrun.utils import logger

if typing.TYPE_CHECKING:
    from mlrun.db.base import RunDBInterface
    from mlrun.projects import MlrunProject
import mlrun.common.schemas.model_monitoring.constants as mm_constants


class _BatchDict(typing.TypedDict):
    minutes: int
    hours: int
    days: int


def get_stream_path(
    project: str = None,
    function_name: str = mm_constants.MonitoringFunctionNames.STREAM,
) -> str:
    """
    Get stream path from the project secret. If wasn't set, take it from the system configurations

    :param project:             Project name.
    :param function_name:    Application name. Default is model_monitoring_stream.

    :return:                    Monitoring stream path to the relevant application.
    """

    stream_uri = mlrun.get_secret_or_env(
        mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
    ) or mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
        target="online",
        function_name=function_name,
    )

    if isinstance(stream_uri, list):  # ML-6043 - user side gets only the new stream uri
        stream_uri = stream_uri[1]  # get new stream path, under projects
    return mlrun.common.model_monitoring.helpers.parse_monitoring_stream_path(
        stream_uri=stream_uri, project=project, function_name=function_name
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


def get_connection_string(secret_provider: typing.Callable[[str], str] = None) -> str:
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


def get_tsdb_connection_string(
    secret_provider: typing.Optional[typing.Callable[[str], str]] = None,
) -> str:
    """Get TSDB connection string from the project secret. If wasn't set, take it from the system
    configurations.
    :param secret_provider: An optional secret provider to get the connection string secret.
    :return:                Valid TSDB connection string.
    """

    return (
        mlrun.get_secret_or_env(
            key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_CONNECTION,
            secret_provider=secret_provider,
        )
        or mlrun.mlconf.model_endpoint_monitoring.tsdb_connection
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
    Get the base period form the controller.

    :param project: Project name.
    :param db:      DB interface.

    :return:    Timedelta for the controller to run.
    :raise:     MLRunNotFoundError if the controller isn't deployed yet
    """

    controller = db.get_function(
        name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        project=project,
    )
    if isinstance(controller, dict):
        controller = mlrun.runtimes.RemoteRuntime.from_dict(controller)
    elif not hasattr(controller, "to_dict"):
        raise mlrun.errors.MLRunNotFoundError()
    base_period = controller.spec.config["spec.triggers.cron_interval"]["attributes"][
        "interval"
    ]
    batch_dict = {
        mm_constants.EventFieldType.MINUTES: int(base_period[:-1]),
        mm_constants.EventFieldType.HOURS: 0,
        mm_constants.EventFieldType.DAYS: 0,
    }
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
        except mlrun.errors.MLRunNotFoundError:
            logger.debug(
                "Not bumping model endpoint last request time - the monitoring controller isn't deployed yet"
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


def calculate_inputs_statistics(
    sample_set_statistics: dict, inputs: pd.DataFrame
) -> mlrun.common.model_monitoring.helpers.FeatureStats:
    """
    Calculate the inputs data statistics for drift monitoring purpose.

    :param sample_set_statistics: The sample set (stored end point's dataset to reference) statistics. The bins of the
                                  histograms of each feature will be used to recalculate the histograms of the inputs.
    :param inputs:                The inputs to calculate their statistics and later on - the drift with respect to the
                                  sample set.

    :returns: The calculated statistics of the inputs data.
    """

    # Use `DFDataInfer` to calculate the statistics over the inputs:
    inputs_statistics = mlrun.data_types.infer.DFDataInfer.get_stats(
        df=inputs,
        options=mlrun.data_types.infer.InferOptions.Histogram,
    )

    # Recalculate the histograms over the bins that are set in the sample-set of the end point:
    for feature in inputs_statistics.keys():
        if feature in sample_set_statistics:
            counts, bins = np.histogram(
                inputs[feature].to_numpy(),
                bins=sample_set_statistics[feature]["hist"][1],
            )
            inputs_statistics[feature]["hist"] = [
                counts.tolist(),
                bins.tolist(),
            ]
        elif "hist" in inputs_statistics[feature]:
            # Comply with the other common features' histogram length
            mlrun.common.model_monitoring.helpers.pad_hist(
                mlrun.common.model_monitoring.helpers.Histogram(
                    inputs_statistics[feature]["hist"]
                )
            )

    return inputs_statistics


def get_endpoint_record(
    project: str,
    endpoint_id: str,
    secret_provider: typing.Optional[typing.Callable[[str], str]] = None,
) -> dict[str, typing.Any]:
    model_endpoint_store = mlrun.model_monitoring.get_store_object(
        project=project, secret_provider=secret_provider
    )
    return model_endpoint_store.get_model_endpoint(endpoint_id=endpoint_id)


def get_result_instance_fqn(
    model_endpoint_id: str, app_name: str, result_name: str
) -> str:
    return f"{model_endpoint_id}.{app_name}.result.{result_name}"


def get_default_result_instance_fqn(model_endpoint_id: str) -> str:
    return get_result_instance_fqn(
        model_endpoint_id,
        mm_constants.HistogramDataDriftApplicationConstants.NAME,
        mm_constants.HistogramDataDriftApplicationConstants.GENERAL_RESULT_NAME,
    )


def get_invocations_fqn(project: str) -> str:
    return _compose_full_name(
        project=project,
        app=mm_constants.SpecialApps.MLRUN_INFRA,
        name=mm_constants.PredictionsQueryConstants.INVOCATIONS,
        type=ModelEndpointMonitoringMetricType.METRIC,
    )


def get_invocations_metric(project: str) -> ModelEndpointMonitoringMetric:
    """
    Return the invocations metric of any model endpoint in the given project.

    :param project: The project name.
    :returns:       The model monitoring metric object.
    """
    return ModelEndpointMonitoringMetric(
        project=project,
        app=mm_constants.SpecialApps.MLRUN_INFRA,
        type=ModelEndpointMonitoringMetricType.METRIC,
        name=mm_constants.PredictionsQueryConstants.INVOCATIONS,
        full_name=get_invocations_fqn(project),
    )
