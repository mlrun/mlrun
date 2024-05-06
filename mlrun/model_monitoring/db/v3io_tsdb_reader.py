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
import mlrun.model_monitoring.writer as mm_writer
import mlrun.utils.v3io_clients


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
    raise NotImplementedError
    return []
