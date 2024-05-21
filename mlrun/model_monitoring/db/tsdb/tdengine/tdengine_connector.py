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
#
import typing

import pandas as pd
import taosws

import mlrun.common.schemas.model_monitoring as mm_constants
import mlrun.model_monitoring.db
import mlrun.model_monitoring.db.tsdb.tdengine.schemas as tdengine_schemas
import mlrun.model_monitoring.db.tsdb.tdengine.stream_graph_steps
from mlrun.utils import logger


class TDEngineConnector(mlrun.model_monitoring.db.TSDBConnector):
    """
    Handles the TSDB operations when the TSDB connector is of type TDEngine.
    """

    def __init__(
        self,
        project: str,
        secret_provider: typing.Callable = None,
        database: str = tdengine_schemas._MODEL_MONITORING_DATABASE,
    ):
        super().__init__(project=project)
        self._tdengine_connection_string = (
            mlrun.model_monitoring.helpers.get_tsdb_connection_string(
                secret_provider=secret_provider
            )
        )
        self.database = database
        self._connection = self._create_connection()
        self._init_super_tables()

    def _create_connection(self):
        """Establish a connection to the TSDB server."""
        conn = taosws.connect(self._tdengine_connection_string)
        try:
            conn.execute(f"CREATE DATABASE {self.database}")
        except taosws.QueryError:
            # Database already exists
            pass
        conn.execute(f"USE {self.database}")
        return conn

    def _init_super_tables(self):
        """Initialize the super tables for the TSDB."""
        self.tables = {
            mm_constants.TDEngineSuperTables.APP_RESULTS: tdengine_schemas.AppResultTable(),
            mm_constants.TDEngineSuperTables.METRICS: tdengine_schemas.Metrics(),
            mm_constants.TDEngineSuperTables.PREDICTIONS: tdengine_schemas.Predictions(),
        }

    def create_tables(self):
        """Create TDEngine supertables."""
        for table in self.tables:
            create_table_query = self.tables[table]._create_super_table_query()
            self._connection.execute(create_table_query)

    def write_application_event(
        self,
        event: dict,
        kind: mm_constants.WriterEventKind = mm_constants.WriterEventKind.RESULT,
    ):
        """
        Write a single result or metric to TSDB.
        """

        table_name = (
            f"{self.project}_"
            f"{event[mm_constants.WriterEvent.ENDPOINT_ID]}_"
            f"{event[mm_constants.WriterEvent.APPLICATION_NAME]}_"
        )
        event[mm_constants.EventFieldType.PROJECT] = self.project

        if kind == mm_constants.WriterEventKind.RESULT:
            # Write a new result
            table = self.tables[mm_constants.TDEngineSuperTables.APP_RESULTS]
            table_name = (
                f"{table_name}_" f"{event[mm_constants.ResultData.RESULT_NAME]}"
            ).replace("-", "_")

        else:
            # Write a new metric
            table = self.tables[mm_constants.TDEngineSuperTables.METRICS]
            table_name = (
                f"{table_name}_" f"{event[mm_constants.MetricData.METRIC_NAME]}"
            ).replace("-", "_")

        create_table_query = table._create_subtable_query(
            subtable=table_name, values=event
        )
        self._connection.execute(create_table_query)
        insert_table_query = table._insert_subtable_query(
            subtable=table_name, values=event
        )
        self._connection.execute(insert_table_query)

    def apply_monitoring_stream_steps(self, graph):
        """
        Apply TSDB steps on the provided monitoring graph. Throughout these steps, the graph stores live data of
        different key metric dictionaries. This data is being used by the monitoring dashboards in
        grafana. At the moment, we store two types of data:
        - prediction latency.
        - custom metrics.
        """

        def apply_process_before_tsdb():
            graph.add_step(
                "mlrun.model_monitoring.db.tsdb.tdengine.stream_graph_steps.ProcessBeforeTDEngine",
                name="ProcessBeforeTDEngine",
                after="MapFeatureNames",
            )

        def apply_tdengine_target(name, after):
            graph.add_step(
                "storey.TDEngineTarget",
                name=name,
                after=after,
                url=self._tdengine_connection_string,
                supertable=mm_constants.TDEngineSuperTables.PREDICTIONS,
                table_col=mm_constants.EventFieldType.TABLE_COLUMN,
                time_col=mm_constants.EventFieldType.TIME,
                database=self.database,
                columns=[
                    mm_constants.EventFieldType.LATENCY,
                    mm_constants.EventKeyMetrics.CUSTOM_METRICS,
                ],
                tag_cols=[
                    mm_constants.EventFieldType.PROJECT,
                    mm_constants.EventFieldType.ENDPOINT_ID,
                ],
                max_events=10,
            )

        apply_process_before_tsdb()
        apply_tdengine_target(
            name="TDEngineTarget",
            after="ProcessBeforeTDEngine",
        )

    def delete_tsdb_resources(self):
        """
        Delete all project resources in the TSDB connector, such as model endpoints data and drift results.
        """
        for table in self.tables:
            get_subtable_names_query = self.tables[table]._get_subtables_query(
                values={mm_constants.EventFieldType.PROJECT: self.project}
            )
            subtables = self._connection.query(get_subtable_names_query)
            for subtable in subtables:
                drop_query = self.tables[table]._drop_subtable_query(
                    subtable=subtable[0]
                )
                self._connection.execute(drop_query)
        logger.info(
            f"Deleted all project resources in the TSDB connector for project {self.project}"
        )

    def get_model_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str,
        end: str,
    ) -> dict[str, list[tuple[str, float]]]:
        # Not implemented, use get_records() instead
        pass

    def get_records(
        self,
        table: str,
        start: str,
        end: str,
        columns: list[str] = None,
        filter_query: str = "",
        timestamp_column: str = mm_constants.EventFieldType.TIME,
    ) -> pd.DataFrame:
        """
        Getting records from TSDB data collection.
        :param table:            Either a supertable or a subtable name.
        :param columns:          Columns to include in the result.
        :param filter_query:     Optional filter expression as a string. The filter structure depends on the TSDB
                                 connector type.
        :param start:            The start time of the metrics.
        :param end:              The end time of the metrics.
        :param timestamp_column: The column name that holds the timestamp.

        :return: DataFrame with the provided attributes from the data collection.
        :raise:  MLRunInvalidArgumentError if query the provided table failed.
        """

        filter_query += f" project = '{self.project}'"

        full_query = tdengine_schemas.TDEngineSchema._get_records_query(
            table=table,
            columns_to_filter=columns,
            filter_query=filter_query,
            start=start,
            end=end,
            timestamp_column=timestamp_column,
            database=self.database,
        )
        try:
            query_result = self._connection.query(full_query)
        except taosws.QueryError as e:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Failed to query table {table} in database {self.database}, {str(e)}"
            )
        columns = []
        for column in query_result.fields:
            columns.append(column.name())

        return pd.DataFrame(query_result, columns=columns)
