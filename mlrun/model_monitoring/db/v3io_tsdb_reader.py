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
from typing import Literal, Union

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
)
from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase
from mlrun.model_monitoring.db.tsdb.v3io.v3io_connector import _TSDB_BE
from mlrun.utils import logger


def _get_sql_query(
    endpoint_id: str,
    names: list[tuple[str, str]],
    table_name: str = mm_constants.MonitoringTSDBTables.APP_RESULTS,
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
                    f"AND {mm_writer.ResultData.RESULT_NAME}='{result_name}')"
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
    type: Literal["metrics", "results"] = "results",
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
    client = mlrun.utils.v3io_clients.get_frames_client(
        address=mlrun.mlconf.v3io_framesd,
        container=KVStoreBase.get_v3io_monitoring_apps_container(project),
    )

    if type == "metrics":
        table_name = mm_constants.MonitoringTSDBTables.METRICS
        df_handler = df_to_metrics_values
    elif type == "results":
        table_name = mm_constants.MonitoringTSDBTables.APP_RESULTS
        df_handler = df_to_results_values
    else:
        raise ValueError(f"Invalid {type = }")

    df: pd.DataFrame = client.read(
        backend=_TSDB_BE,
        query=_get_sql_query(
            endpoint_id,
            [(metric.app, metric.name) for metric in metrics],
            table_name=table_name,
        ),
        start=start,
        end=end,
    )

    return df_handler(df=df, metrics=metrics, project=project)


def df_to_results_values(
    *, df: pd.DataFrame, metrics: list[ModelEndpointMonitoringMetric], project: str
) -> list[
    Union[ModelEndpointMonitoringResultValues, ModelEndpointMonitoringMetricNoData]
]:
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
    for (app_name, result_name), sub_df in grouped:
        result_kind = _get_result_kind(sub_df)
        full_name = _compose_full_name(project=project, app=app_name, name=result_name)
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
        grouped = []
    for (app_name, result_name), sub_df in grouped:
        full_name = _compose_full_name(project=project, app=app_name, name=result_name)
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
