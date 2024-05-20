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

# TODO: Move this module into the TSDB abstraction once it is in.

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
    ModelEndpointMonitoringMetricType,
    ModelEndpointMonitoringResultNoData,
    ModelEndpointMonitoringResultValues,
    _compose_full_name,
    _ModelEndpointMonitoringResultValuesBase,
)
from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase
from mlrun.model_monitoring.db.tsdb.v3io.v3io_connector import _TSDB_BE
from mlrun.utils import logger


def _get_sql_query(endpoint_id: str, names: list[tuple[str, str]]) -> str:
    with StringIO() as query:
        query.write(
            f"SELECT * FROM '{mm_constants.MonitoringTSDBTables.APP_RESULTS}' "
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


def read_data(
    *,
    project: str,
    endpoint_id: str,
    start: datetime,
    end: datetime,
    metrics: list[ModelEndpointMonitoringMetric],
) -> list[_ModelEndpointMonitoringResultValuesBase]:
    client = mlrun.utils.v3io_clients.get_frames_client(
        address=mlrun.mlconf.v3io_framesd,
        container=KVStoreBase.get_v3io_monitoring_apps_container(project),
    )
    df: pd.DataFrame = client.read(
        backend=_TSDB_BE,
        query=_get_sql_query(
            endpoint_id, [(metric.app, metric.name) for metric in metrics]
        ),
        start=start,
        end=end,
    )

    metrics_without_data = {metric.full_name: metric for metric in metrics}

    metrics_values: list[_ModelEndpointMonitoringResultValuesBase] = []
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
                type=ModelEndpointMonitoringMetricType.RESULT,
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
            ModelEndpointMonitoringResultNoData(
                full_name=metric.full_name,
                type=ModelEndpointMonitoringMetricType.RESULT,
            )
        )

    return metrics_values


def read_predictions(
    *,
    project: str,
    endpoint_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    aggregation_window: Optional[str] = None,
    limit: Optional[int] = None,
) -> Union[
    mlrun.common.schemas.model_monitoring.model_endpoints.ModelEndpointMonitoringResultValues,
    mlrun.common.schemas.model_monitoring.model_endpoints.ModelEndpointMonitoringResultNoData,
]:
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

    full_name = (
        mlrun.common.schemas.model_monitoring.model_endpoints._compose_full_name(
            project=project,
            app="mlrun-infra",
            name=mlrun.common.schemas.model_monitoring.MonitoringTSDBTables.INVOCATIONS,
        )
    )

    if df.empty:
        return ModelEndpointMonitoringResultNoData(
            full_name=full_name,
            type=ModelEndpointMonitoringMetricType.METRIC,
        )

    rows = df.reset_index().to_dict(orient="records")
    values = [
        [
            row["time"],
            row["count(latency)"],  # event count for the time window
            mm_constants.ResultStatusApp.irrelevant,
        ]
        for row in rows
    ]
    return ModelEndpointMonitoringResultValues(
        full_name=full_name,
        type=ModelEndpointMonitoringMetricType.METRIC,
        result_kind=mm_constants.ResultKindApp.system_performance,
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
        limit=1,  # Read just one record, because we just want to check if there is any data for this endpoint_id
    )
    if predictions:
        return ModelEndpointMonitoringMetric(
            project=project,
            app="mlrun-infra",
            type=ModelEndpointMonitoringMetricType.METRIC,
            name=mlrun.common.schemas.model_monitoring.MonitoringTSDBTables.INVOCATIONS,
            full_name=mlrun.common.schemas.model_monitoring.model_endpoints._compose_full_name(
                project=project,
                app="mlrun-infra",
                name=mlrun.common.schemas.model_monitoring.MonitoringTSDBTables.INVOCATIONS,
            ),
        )
