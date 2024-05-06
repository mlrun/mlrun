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

import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.writer as mm_writer
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpointMonitoringMetricType,
    ModelEndpointMonitoringResultValues,
    _compose_full_name,
)
from mlrun.utils import logger


def _get_sql_query(endpoint_id: str, names: list[tuple[str, str]]) -> str:
    query = (
        f"SELECT * FROM '{mm_writer._TSDB_TABLE}' "
        f"WHERE {mm_writer.WriterEvent.ENDPOINT_ID}='{endpoint_id}'"
    )

    if names:
        query += " AND ("

        for i, (app_name, result_name) in enumerate(names):
            sub_cond = (
                f"({mm_writer.WriterEvent.APPLICATION_NAME}='{app_name}' "
                f"AND {mm_writer.ResultData.RESULT_NAME}='{result_name}')"
            )
            if i != 0:  # not first sub condition
                query += " OR "
            query += sub_cond

        query += ")"

    query += ";"
    return query


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
    kind_str = unique_kinds[0]
    return getattr(mm_constants.ResultKindApp, kind_str.split(".")[1])


def read_data(
    project: str,
    endpoint_id: str,
    start: datetime,
    end: datetime,
    names: list[tuple[str, str]],
) -> list:
    client = mlrun.utils.v3io_clients.get_frames_client(
        address=mlrun.mlconf.v3io_framesd,
        container=mm_writer.ModelMonitoringWriter.get_v3io_container(project),
    )
    df: pd.DataFrame = client.read(
        backend=mm_writer._TSDB_BE,
        query=_get_sql_query(endpoint_id, names),
        start=start,
        end=end,
    )
    grouped = df.groupby(
        [mm_writer.WriterEvent.APPLICATION_NAME, mm_writer.ResultData.RESULT_NAME],
        observed=False,
    )
    metrics_values: list[ModelEndpointMonitoringResultValues] = []
    for (app_name, result_name), sub_df in grouped:
        result_kind = _get_result_kind(sub_df)
        metrics_values.append(
            ModelEndpointMonitoringResultValues(
                full_name=_compose_full_name(
                    project=project, app=app_name, name=result_name
                ),
                type=ModelEndpointMonitoringMetricType.RESULT,
                result_kind=result_kind,
                values=list(
                    zip(
                        list(sub_df.index),
                        list(sub_df[mm_writer.ResultData.RESULT_VALUE]),
                        list(sub_df[mm_writer.ResultData.RESULT_STATUS]),
                    )
                ),  # pyright: ignore[reportArgumentType]
            )
        )
    return metrics_values
