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

# TODO: Move this module into the TSDB abstraction:
# mlrun/model_monitoring/db/tsdb/v3io/v3io_connector.py

from datetime import datetime
from io import StringIO
from typing import Optional, Union

import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.writer as mm_writer
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricNoData,
    ModelEndpointMonitoringMetricType,
    ModelEndpointMonitoringMetricValues,
    ModelEndpointMonitoringResultValues,
    _compose_full_name,
    _ModelEndpointMonitoringMetricValuesBase,
)
from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase
from mlrun.model_monitoring.db.tsdb.v3io.v3io_connector import _TSDB_BE
from mlrun.utils import logger


def _get_sql_query(
    endpoint_id: str,
    names: list[tuple[str, str]],
    table_name: str = mm_constants.MonitoringTSDBTables.APP_RESULTS,
    name: str = mm_writer.ResultData.RESULT_NAME,
) -> str:
    with StringIO() as query:
        query.write(
            f"SELECT * FROM '{table_name}' "
            f"WHERE {mm_writer.WriterEvent.ENDPOINT_ID}='{endpoint_id}'"
        )
        if names:
            query.write(" AND (")

            for i, (app_name, result_name) in enumerate(names):
                sub_cond = (
                    f"({mm_writer.WriterEvent.APPLICATION_NAME}='{app_name}' "
                    f"AND {name}='{result_name}')"
                )
                if i != 0:  # not first sub condition
                    query.write(" OR ")
                query.write(sub_cond)

            query.write(")")

        query.write(";")
        return query.getvalue()


def _get_result_kind(result_df: pd.DataFrame) -> mm_constants.ResultKindApp:
    kind_series = result_df[mm_writer.ResultData.RESULT_KIND]
    unique_kinds = kind_series.unique()
    if len(unique_kinds) > 1:
        logger.warning(
            "The result has more than one kind",
            kinds=list(unique_kinds),
            application_name=result_df[mm_writer.WriterEvent.APPLICATION_NAME],
            result_name=result_df[mm_writer.ResultData.RESULT_NAME],
        )
    return unique_kinds[0]


def read_metrics_data(
    *,
    project: str,
    endpoint_id: str,
    start: datetime,
    end: datetime,
    metrics: list[ModelEndpointMonitoringMetric],
    type: str = "results",
) -> Union[
    list[
        Union[
            ModelEndpointMonitoringResultValues,
            ModelEndpointMonitoringMetricNoData,
        ],
    ],
    list[
        Union[
            ModelEndpointMonitoringMetricValues,
            ModelEndpointMonitoringMetricNoData,
        ],
    ],
]:
    """
    Read metrics OR results from the TSDB and return as a list.
    Note: the type must match the actual metrics in the `metrics` parameter.
    If the type is "results", pass only results in the `metrics` parameter.
    """
    client = mlrun.utils.v3io_clients.get_frames_client(
        address=mlrun.mlconf.v3io_framesd,
        container=KVStoreBase.get_v3io_monitoring_apps_container(project),
    )

    if type == "metrics":
        table_name = mm_constants.MonitoringTSDBTables.METRICS
        name = mm_constants.MetricData.METRIC_NAME
        df_handler = df_to_metrics_values
    elif type == "results":
        table_name = mm_constants.MonitoringTSDBTables.APP_RESULTS
        name = mm_constants.ResultData.RESULT_NAME
        df_handler = df_to_results_values
    else:
        raise ValueError(f"Invalid {type = }")

    query = _get_sql_query(
        endpoint_id,
        [(metric.app, metric.name) for metric in metrics],
        table_name=table_name,
        name=name,
    )

    logger.debug("Querying V3IO TSDB", query=query)

    df: pd.DataFrame = client.read(
        backend=_TSDB_BE,
        query=query,
        start=start,
        end=end,
    )

    logger.debug(
        "Read a data-frame", project=project, endpoint_id=endpoint_id, is_empty=df.empty
    )

    return df_handler(df=df, metrics=metrics, project=project)


def df_to_results_values(
    *, df: pd.DataFrame, metrics: list[ModelEndpointMonitoringMetric], project: str
) -> list[
    Union[ModelEndpointMonitoringResultValues, ModelEndpointMonitoringMetricNoData]
]:
    """
    Parse a time-indexed data-frame of results from the TSDB into a list of
    results values per distinct results.
    When a result is not found in the data-frame, it is represented in no-data object.
    """
    metrics_without_data = {metric.full_name: metric for metric in metrics}

    metrics_values: list[
        Union[ModelEndpointMonitoringResultValues, ModelEndpointMonitoringMetricNoData]
    ] = []
    if not df.empty:
        grouped = df.groupby(
            [mm_writer.WriterEvent.APPLICATION_NAME, mm_writer.ResultData.RESULT_NAME],
            observed=False,
        )
    else:
        grouped = []
        logger.debug("No results", missing_results=metrics_without_data.keys())
    for (app_name, name), sub_df in grouped:
        result_kind = _get_result_kind(sub_df)
        full_name = _compose_full_name(project=project, app=app_name, name=name)
        metrics_values.append(
            ModelEndpointMonitoringResultValues(
                full_name=full_name,
                result_kind=result_kind,
                values=list(
                    zip(
                        sub_df.index,
                        sub_df[mm_writer.ResultData.RESULT_VALUE],
                        sub_df[mm_writer.ResultData.RESULT_STATUS],
                    )
                ),  # pyright: ignore[reportArgumentType]
            )
        )
        del metrics_without_data[full_name]

    for metric in metrics_without_data.values():
        if metric.full_name == get_invocations_fqn(project):
            continue
        metrics_values.append(
            ModelEndpointMonitoringMetricNoData(
                full_name=metric.full_name,
                type=ModelEndpointMonitoringMetricType.RESULT,
            )
        )

    return metrics_values


def df_to_metrics_values(
    *, df: pd.DataFrame, metrics: list[ModelEndpointMonitoringMetric], project: str
) -> list[
    Union[ModelEndpointMonitoringMetricValues, ModelEndpointMonitoringMetricNoData]
]:
    """
    Parse a time-indexed data-frame of metrics from the TSDB into a list of
    metrics values per distinct results.
    When a metric is not found in the data-frame, it is represented in no-data object.
    """
    metrics_without_data = {metric.full_name: metric for metric in metrics}

    metrics_values: list[
        Union[ModelEndpointMonitoringMetricValues, ModelEndpointMonitoringMetricNoData]
    ] = []
    if not df.empty:
        grouped = df.groupby(
            [mm_writer.WriterEvent.APPLICATION_NAME, mm_writer.MetricData.METRIC_NAME],
            observed=False,
        )
    else:
        logger.debug("No metrics", missing_metrics=metrics_without_data.keys())
        grouped = []
    for (app_name, name), sub_df in grouped:
        full_name = _compose_full_name(
            project=project,
            app=app_name,
            name=name,
            type=ModelEndpointMonitoringMetricType.METRIC,
        )
        metrics_values.append(
            ModelEndpointMonitoringMetricValues(
                full_name=full_name,
                values=list(
                    zip(
                        sub_df.index,
                        sub_df[mm_writer.MetricData.METRIC_VALUE],
                    )
                ),  # pyright: ignore[reportArgumentType]
            )
        )
        del metrics_without_data[full_name]

    for metric in metrics_without_data.values():
        metrics_values.append(
            ModelEndpointMonitoringMetricNoData(
                full_name=metric.full_name,
                type=ModelEndpointMonitoringMetricType.METRIC,
            )
        )

    return metrics_values


def get_invocations_fqn(project: str):
    return mlrun.common.schemas.model_monitoring.model_endpoints._compose_full_name(
        project=project,
        app="mlrun-infra",
        name=mlrun.common.schemas.model_monitoring.MonitoringTSDBTables.INVOCATIONS,
        type=mlrun.common.schemas.model_monitoring.ModelEndpointMonitoringMetricType.METRIC,
    )


def read_predictions(
    *,
    project: str,
    endpoint_id: str,
    start: Optional[Union[datetime, str]] = None,
    end: Optional[Union[datetime, str]] = None,
    aggregation_window: Optional[str] = None,
    limit: Optional[int] = None,
) -> _ModelEndpointMonitoringMetricValuesBase:
    client = mlrun.utils.v3io_clients.get_frames_client(
        address=mlrun.mlconf.v3io_framesd,
        container="users",
    )
    frames_client_kwargs = {}
    if aggregation_window:
        frames_client_kwargs["step"] = aggregation_window
        frames_client_kwargs["aggregation_window"] = aggregation_window
    if limit:
        frames_client_kwargs["limit"] = limit
    df: pd.DataFrame = client.read(
        backend=_TSDB_BE,
        table=f"pipelines/{project}/model-endpoints/predictions",
        columns=["latency"],
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
        aggregators="count",
        **frames_client_kwargs,
    )

    full_name = get_invocations_fqn(project)

    if df.empty:
        return ModelEndpointMonitoringMetricNoData(
            full_name=full_name,
            type=ModelEndpointMonitoringMetricType.METRIC,
        )

    rows = df.reset_index().to_dict(orient="records")
    values = [
        [
            row["time"],
            row["count(latency)"],  # event count for the time window
        ]
        for row in rows
    ]
    return ModelEndpointMonitoringMetricValues(
        full_name=full_name,
        values=values,
    )


def read_prediction_metric_for_endpoint_if_exists(
    *,
    project: str,
    endpoint_id: str,
) -> Optional[ModelEndpointMonitoringMetric]:
    predictions = read_predictions(
        project=project,
        endpoint_id=endpoint_id,
        start="0",
        end="now",
        limit=1,  # Read just one record, because we just want to check if there is any data for this endpoint_id
    )
    if predictions:
        return ModelEndpointMonitoringMetric(
            project=project,
            app="mlrun-infra",
            type=ModelEndpointMonitoringMetricType.METRIC,
            name=mlrun.common.schemas.model_monitoring.MonitoringTSDBTables.INVOCATIONS,
            full_name=get_invocations_fqn(project),
        )
