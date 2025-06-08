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

import threading
from datetime import datetime, timedelta
from typing import Callable, Final, Literal, Optional, Union

import pandas as pd
import taosws

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.common.types
import mlrun.model_monitoring.db.tsdb.tdengine.schemas as tdengine_schemas
import mlrun.model_monitoring.db.tsdb.tdengine.stream_graph_steps
from mlrun.datastore.datastore_profile import DatastoreProfile
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.db.tsdb.tdengine.tdengine_connection import (
    Statement,
    TDEngineConnection,
)
from mlrun.model_monitoring.helpers import get_invocations_fqn
from mlrun.utils import logger

# Thread-local storage for connections
_thread_local = threading.local()


class TDEngineTimestampPrecision(mlrun.common.types.StrEnum):
    """
    The timestamp precision for the TDEngine database.
    For more information, see:
    https://docs.tdengine.com/tdengine-reference/sql-manual/data-types/#timestamp
    https://docs.tdengine.com/tdengine-reference/sql-manual/manage-databases/#create-database
    """

    MILLISECOND = "ms"  # TDEngine's default
    MICROSECOND = "us"  # MLRun's default
    NANOSECOND = "ns"


class TDEngineConnector(TSDBConnector):
    """
    Handles the TSDB operations when the TSDB connector is of type TDEngine.
    """

    type: str = mm_schemas.TSDBTarget.TDEngine
    database = f"{tdengine_schemas._MODEL_MONITORING_DATABASE}_{mlrun.mlconf.system_id}"

    def __init__(
        self,
        project: str,
        profile: DatastoreProfile,
        timestamp_precision: TDEngineTimestampPrecision = TDEngineTimestampPrecision.MICROSECOND,
        **kwargs,
    ):
        super().__init__(project=project)

        self._tdengine_connection_profile = profile

        self._timestamp_precision: Final = (  # cannot be changed after initialization
            timestamp_precision
        )

        self._init_super_tables()

    @property
    def connection(self) -> TDEngineConnection:
        if not hasattr(_thread_local, "connection"):
            _thread_local.connection = self._create_connection()
            logger.debug(
                "Created new TDEngine connection for thread",
                project=self.project,
                thread_name=threading.current_thread().name,
                thread_id=threading.get_ident(),
            )
        return _thread_local.connection

    def _create_connection(self) -> TDEngineConnection:
        """Establish a connection to the TSDB server."""
        logger.debug("Creating a new connection to TDEngine", project=self.project)
        conn = TDEngineConnection(
            self._tdengine_connection_profile.dsn(),
        )
        conn.prefix_statements = [f"USE {self.database}"]

        return conn

    def _init_super_tables(self):
        """Initialize the super tables for the TSDB."""
        self.tables = {
            mm_schemas.TDEngineSuperTables.APP_RESULTS: tdengine_schemas.AppResultTable(
                project=self.project, database=self.database
            ),
            mm_schemas.TDEngineSuperTables.METRICS: tdengine_schemas.Metrics(
                project=self.project, database=self.database
            ),
            mm_schemas.TDEngineSuperTables.PREDICTIONS: tdengine_schemas.Predictions(
                project=self.project, database=self.database
            ),
            mm_schemas.TDEngineSuperTables.ERRORS: tdengine_schemas.Errors(
                project=self.project, database=self.database
            ),
        }

    def _create_db_if_not_exists(self):
        """Create the database if it does not exist."""
        self.connection.prefix_statements = []
        self.connection.run(
            statements=f"CREATE DATABASE IF NOT EXISTS {self.database} PRECISION '{self._timestamp_precision}'",
        )
        self.connection.prefix_statements = [f"USE {self.database}"]
        logger.debug(
            "The TDEngine database is currently in use",
            project=self.project,
            database=self.database,
        )

    def create_tables(self):
        """Create TDEngine supertables."""

        # Create the database if it does not exist
        self._create_db_if_not_exists()

        for table in self.tables:
            create_table_query = self.tables[table]._create_super_table_query()
            conn = self.connection
            conn.run(
                statements=create_table_query,
            )

    def write_application_event(
        self,
        event: dict,
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a single result or metric to TSDB.
        """

        table_name = (
            f"{event[mm_schemas.WriterEvent.ENDPOINT_ID]}_"
            f"{event[mm_schemas.WriterEvent.APPLICATION_NAME]}"
        )

        if kind == mm_schemas.WriterEventKind.RESULT:
            # Write a new result
            table = self.tables[mm_schemas.TDEngineSuperTables.APP_RESULTS]
            table_name = (
                f"{table_name}_{event[mm_schemas.ResultData.RESULT_NAME]}"
            ).replace("-", "_")

        else:
            # Write a new metric
            table = self.tables[mm_schemas.TDEngineSuperTables.METRICS]
            table_name = (
                f"{table_name}_{event[mm_schemas.MetricData.METRIC_NAME]}"
            ).replace("-", "_")

        # Escape the table name for case-sensitivity (ML-7908)
        # https://github.com/taosdata/taos-connector-python/issues/260
        table_name = f"`{table_name}`"

        # Convert the datetime strings to datetime objects
        event[mm_schemas.WriterEvent.END_INFER_TIME] = self._convert_to_datetime(
            val=event[mm_schemas.WriterEvent.END_INFER_TIME]
        )
        event[mm_schemas.WriterEvent.START_INFER_TIME] = self._convert_to_datetime(
            val=event[mm_schemas.WriterEvent.START_INFER_TIME]
        )

        create_table_sql = table._create_subtable_sql(subtable=table_name, values=event)

        # we need the string values to be sent to the connection, not the enum
        columns = {str(key): str(val) for key, val in table.columns.items()}

        insert_statement = Statement(
            columns=columns,
            subtable=table_name,
            values=event,
            timestamp_precision=self._timestamp_precision,
        )

        self.connection.run(
            statements=[
                create_table_sql,
                insert_statement,
            ],
        )

    @staticmethod
    def _convert_to_datetime(val: Union[str, datetime]) -> datetime:
        return datetime.fromisoformat(val) if isinstance(val, str) else val

    @staticmethod
    def _get_endpoint_filter(endpoint_id: Union[str, list[str]]) -> str:
        if isinstance(endpoint_id, str):
            return f"endpoint_id='{endpoint_id}'"
        elif isinstance(endpoint_id, list):
            return f"endpoint_id IN({str(endpoint_id)[1:-1]}) "
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid 'endpoint_id' filter: must be a string or a list."
            )

    def _drop_database_query(self) -> str:
        return f"DROP DATABASE IF EXISTS {self.database};"

    def _get_table_name_query(self) -> str:
        return f"SELECT table_name FROM information_schema.ins_tables where db_name='{self.database}' LIMIT 1;"

    def apply_monitoring_stream_steps(self, graph, **kwarg):
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
                after="FilterNOP",
            )

        def apply_tdengine_target(name, after):
            graph.add_step(
                "mlrun.datastore.storeytargets.TDEngineStoreyTarget",
                name=name,
                after=after,
                url=f"ds://{self._tdengine_connection_profile.name}",
                supertable=self.tables[
                    mm_schemas.TDEngineSuperTables.PREDICTIONS
                ].super_table,
                table_col=mm_schemas.EventFieldType.TABLE_COLUMN,
                time_col=mm_schemas.EventFieldType.TIME,
                database=self.database,
                columns=[
                    mm_schemas.EventFieldType.LATENCY,
                    mm_schemas.EventKeyMetrics.CUSTOM_METRICS,
                    mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT,
                    mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT,
                ],
                tag_cols=[
                    mm_schemas.EventFieldType.ENDPOINT_ID,
                ],
                max_events=1000,
                flush_after_seconds=30,
            )

        apply_process_before_tsdb()
        apply_tdengine_target(
            name="TDEngineTarget",
            after="ProcessBeforeTDEngine",
        )

    def handle_model_error(
        self,
        graph,
        tsdb_batching_max_events: int = 1000,
        tsdb_batching_timeout_secs: int = 30,
        **kwargs,
    ) -> None:
        graph.add_step(
            "mlrun.model_monitoring.db.tsdb.tdengine.stream_graph_steps.ErrorExtractor",
            name="error_extractor",
            after="ForwardError",
        )
        graph.add_step(
            "mlrun.datastore.storeytargets.TDEngineStoreyTarget",
            name="tsdb_error",
            after="error_extractor",
            url=f"ds://{self._tdengine_connection_profile.name}",
            supertable=self.tables[mm_schemas.TDEngineSuperTables.ERRORS].super_table,
            table_col=mm_schemas.EventFieldType.TABLE_COLUMN,
            time_col=mm_schemas.EventFieldType.TIME,
            database=self.database,
            columns=[
                mm_schemas.EventFieldType.MODEL_ERROR,
            ],
            tag_cols=[
                mm_schemas.EventFieldType.ENDPOINT_ID,
                mm_schemas.EventFieldType.ERROR_TYPE,
            ],
            max_events=tsdb_batching_max_events,
            flush_after_seconds=tsdb_batching_timeout_secs,
        )

    def delete_tsdb_records(
        self,
        endpoint_ids: list[str],
    ):
        """
        To delete subtables within TDEngine, we first query the subtables names with the provided endpoint_ids.
        Then, we drop each subtable.
        """
        logger.debug(
            "Deleting model endpoint resources using the TDEngine connector",
            project=self.project,
            number_of_endpoints_to_delete=len(endpoint_ids),
        )

        # Get all subtables with the provided endpoint_ids
        subtables = []
        try:
            for table in self.tables:
                get_subtable_query = self.tables[table]._get_subtables_query_by_tag(
                    filter_tag="endpoint_id", filter_values=endpoint_ids
                )
                subtables_result = self.connection.run(
                    query=get_subtable_query,
                )
                subtables.extend([subtable[0] for subtable in subtables_result.data])
        except Exception as e:
            logger.warning(
                "Failed to get subtables for deletion. You may need to delete them manually."
                "These can be found under the following supertables: app_results, "
                "metrics, errors, and predictions.",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )

        # Prepare the drop statements
        drop_statements = []
        for subtable in subtables:
            drop_statements.append(
                self.tables[table].drop_subtable_query(subtable=subtable)
            )
        try:
            self.connection.run(
                statements=drop_statements,
            )
        except Exception as e:
            logger.warning(
                "Failed to delete model endpoint resources. You may need to delete them manually. "
                "These can be found under the following supertables: app_results, "
                "metrics, errors, and predictions.",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )
        logger.debug(
            "Deleted all model endpoint resources using the TDEngine connector",
            project=self.project,
            number_of_endpoints_to_delete=len(endpoint_ids),
        )

    def delete_tsdb_resources(self):
        """
        Delete all project resources in the TSDB connector, such as model endpoints data and drift results.
        """
        logger.debug(
            "Deleting all project resources using the TDEngine connector",
            project=self.project,
        )
        drop_statements = []
        for table in self.tables:
            drop_statements.append(self.tables[table].drop_supertable_query())

        try:
            self.connection.run(
                statements=drop_statements,
            )
        except Exception as e:
            logger.warning(
                "Failed to drop TDEngine tables. You may need to drop them manually. "
                "These can be found under the following supertables: app_results, "
                "metrics, errors, and predictions.",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )
        logger.debug(
            "Deleted all project resources using the TDEngine connector",
            project=self.project,
        )

        # Check if database is empty and if so, drop it
        self._drop_database_if_empty()

    def _drop_database_if_empty(self):
        query_random_table_name = self._get_table_name_query()
        drop_database = False
        try:
            table_name = self.connection.run(
                query=query_random_table_name,
            )
            if len(table_name.data) == 0:
                # no tables were found under the database
                drop_database = True

        except Exception as e:
            logger.warning(
                "Failed to query tables in the database. You may need to drop the database manually if it is empty.",
                project=self.project,
                error=mlrun.errors.err_to_str(e),
            )

        if drop_database:
            logger.debug(
                "Going to drop the TDEngine database",
                project=self.project,
                database=self.database,
            )
            drop_database_query = self._drop_database_query()
            try:
                self.connection.run(
                    statements=drop_database_query,
                )
                logger.debug(
                    "The TDEngine database has been successfully dropped",
                    project=self.project,
                    database=self.database,
                )

            except Exception as e:
                logger.warning(
                    "Failed to drop the database. You may need to drop it manually if it is empty.",
                    project=self.project,
                    error=mlrun.errors.err_to_str(e),
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

    def _get_records(
        self,
        table: str,
        start: datetime,
        end: datetime,
        columns: Optional[list[str]] = None,
        filter_query: Optional[str] = None,
        interval: Optional[str] = None,
        agg_funcs: Optional[list] = None,
        limit: Optional[int] = None,
        sliding_window_step: Optional[str] = None,
        timestamp_column: str = mm_schemas.EventFieldType.TIME,
        group_by: Optional[Union[list[str], str]] = None,
        preform_agg_columns: Optional[list] = None,
        order_by: Optional[str] = None,
        desc: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Getting records from TSDB data collection.
        :param table:                 Either a supertable or a subtable name.
        :param start:                 The start time of the metrics.
        :param end:                   The end time of the metrics.
        :param columns:               Columns to include in the result.
        :param filter_query:          Optional filter expression as a string. TDengine supports SQL-like syntax.
        :param interval:              The interval to aggregate the data by. Note that if interval is provided,
                                      `agg_funcs` must bg provided as well. Provided as a string in the format of '1m',
                                      '1h', etc.
        :param agg_funcs:             The aggregation functions to apply on the columns. Note that if `agg_funcs` is
                                      provided, `interval` must bg provided as well. Provided as a list of strings in
                                      the format of ['sum', 'avg', 'count', ...].
        :param limit:                 The maximum number of records to return.
        :param sliding_window_step:   The time step for which the time window moves forward. Note that if
                                      `sliding_window_step` is provided, interval must be provided as well. Provided
                                      as a string in the format of '1m', '1h', etc.
        :param timestamp_column:      The column name that holds the timestamp index.
        :param group_by:              The column name to group by. Note that if `group_by` is provided, aggregation
                                      functions must bg provided
        :param preform_agg_columns:   The columns to preform aggregation on.
                                      notice that all aggregation functions provided will preform on those columns.
                                      If not provided The default behavior is to preform on all columns in columns,
                                      if an empty list was provided The aggregation won't be performed.
        :param order_by:              The column or alias to preform ordering on the query.
        :param desc:                  Whether or not to sort the results in descending order.

        :return: DataFrame with the provided attributes from the data collection.
        :raise:  MLRunInvalidArgumentError if query the provided table failed.
        """

        full_query = tdengine_schemas.TDEngineSchema._get_records_query(
            table=table,
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            interval=interval,
            limit=limit,
            agg_funcs=agg_funcs,
            sliding_window_step=sliding_window_step,
            timestamp_column=timestamp_column,
            database=self.database,
            group_by=group_by,
            preform_agg_funcs_columns=preform_agg_columns,
            order_by=order_by,
            desc=desc,
        )
        logger.debug("Querying TDEngine", query=full_query)
        try:
            query_result = self.connection.run(
                query=full_query,
            )
        except taosws.QueryError as e:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Failed to query table {table} in database {self.database}, {str(e)}"
            )

        df_columns = [field.name for field in query_result.fields]
        return pd.DataFrame(query_result.data, columns=df_columns)

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: Literal["metrics", "results"],
        with_result_extra_data: bool = False,
    ) -> Union[
        list[
            Union[
                mm_schemas.ModelEndpointMonitoringResultValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
        list[
            Union[
                mm_schemas.ModelEndpointMonitoringMetricValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
    ]:
        timestamp_column = mm_schemas.WriterEvent.END_INFER_TIME
        columns = [timestamp_column, mm_schemas.WriterEvent.APPLICATION_NAME]
        if type == "metrics":
            if with_result_extra_data:
                logger.warning(
                    "The 'with_result_extra_data' parameter is not supported for metrics, just for results",
                    project=self.project,
                    endpoint_id=endpoint_id,
                )
            table = self.tables[mm_schemas.TDEngineSuperTables.METRICS].super_table
            name = mm_schemas.MetricData.METRIC_NAME
            columns += [name, mm_schemas.MetricData.METRIC_VALUE]
            df_handler = self.df_to_metrics_values
        elif type == "results":
            table = self.tables[mm_schemas.TDEngineSuperTables.APP_RESULTS].super_table
            name = mm_schemas.ResultData.RESULT_NAME
            columns += [
                name,
                mm_schemas.ResultData.RESULT_VALUE,
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.ResultData.RESULT_KIND,
            ]
            if with_result_extra_data:
                columns.append(mm_schemas.ResultData.RESULT_EXTRA_DATA)
            df_handler = self.df_to_results_values
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid type {type}, must be either 'metrics' or 'results'."
            )

        metrics_condition = " OR ".join(
            [
                f"({mm_schemas.WriterEvent.APPLICATION_NAME}='{metric.app}' AND {name}='{metric.name}')"
                for metric in metrics
            ]
        )
        filter_query = f"(endpoint_id='{endpoint_id}') AND ({metrics_condition})"

        df = self._get_records(
            table=table,
            start=start,
            end=end,
            filter_query=filter_query,
            timestamp_column=timestamp_column,
            columns=columns,
        )

        df[mm_schemas.WriterEvent.END_INFER_TIME] = pd.to_datetime(
            df[mm_schemas.WriterEvent.END_INFER_TIME]
        )
        df.set_index(mm_schemas.WriterEvent.END_INFER_TIME, inplace=True)

        logger.debug(
            "Converting a DataFrame to a list of metrics or results values",
            table=table,
            project=self.project,
            endpoint_id=endpoint_id,
            is_empty=df.empty,
        )

        if not with_result_extra_data and type == "results":
            # Set the extra data to an empty string if it's not requested
            df[mm_schemas.ResultData.RESULT_EXTRA_DATA] = ""

        return df_handler(df=df, metrics=metrics, project=self.project)

    def read_predictions(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        aggregation_window: Optional[str] = None,
        agg_funcs: Optional[list] = None,
        limit: Optional[int] = None,
    ) -> Union[
        mm_schemas.ModelEndpointMonitoringMetricValues,
        mm_schemas.ModelEndpointMonitoringMetricNoData,
    ]:
        if (agg_funcs and not aggregation_window) or (
            aggregation_window and not agg_funcs
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "both or neither of `aggregation_window` and `agg_funcs` must be provided"
            )
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].super_table,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT],
            filter_query=f"endpoint_id='{endpoint_id}'",
            agg_funcs=agg_funcs,
            interval=aggregation_window,
            limit=limit,
        )

        full_name = get_invocations_fqn(self.project)

        if df.empty:
            return mm_schemas.ModelEndpointMonitoringMetricNoData(
                full_name=full_name,
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )

        if aggregation_window:
            # _wend column, which represents the end time of each window, will be used as the time index
            df["_wend"] = pd.to_datetime(df["_wend"])
            df.set_index("_wend", inplace=True)

        estimated_prediction_count = (
            f"{agg_funcs[0]}({mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT})"
            if agg_funcs
            else mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT
        )

        return mm_schemas.ModelEndpointMonitoringMetricValues(
            full_name=full_name,
            values=list(
                zip(
                    df.index,
                    df[estimated_prediction_count],
                )
            ),  # pyright: ignore[reportArgumentType]
        )

    def get_last_request(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_ids)
        start, end = self._get_start_end(start, end)
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].super_table,
            start=start,
            end=end,
            columns=[
                mm_schemas.EventFieldType.ENDPOINT_ID,
                mm_schemas.EventFieldType.TIME,
                mm_schemas.EventFieldType.LATENCY,
            ],
            filter_query=filter_query,
            timestamp_column=mm_schemas.EventFieldType.TIME,
            agg_funcs=["last"],
            group_by=mm_schemas.EventFieldType.ENDPOINT_ID,
            preform_agg_columns=[mm_schemas.EventFieldType.TIME],
        )
        if not df.empty:
            df.dropna(inplace=True)
        df.rename(
            columns={
                f"last({mm_schemas.EventFieldType.TIME})": mm_schemas.EventFieldType.LAST_REQUEST,
                f"{mm_schemas.EventFieldType.LATENCY}": "last_latency",
            },
            inplace=True,
        )
        df[mm_schemas.EventFieldType.LAST_REQUEST] = pd.to_datetime(
            df[mm_schemas.EventFieldType.LAST_REQUEST],
            errors="coerce",
            format="ISO8601",
            utc=True,
        )
        return df

    def get_drift_status(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> pd.DataFrame:
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_ids)
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._get_start_end(start, end)
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.APP_RESULTS].super_table,
            start=start,
            end=end,
            columns=[
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            filter_query=filter_query,
            timestamp_column=mm_schemas.WriterEvent.END_INFER_TIME,
            agg_funcs=["max"],
            group_by=mm_schemas.EventFieldType.ENDPOINT_ID,
            preform_agg_columns=[mm_schemas.ResultData.RESULT_STATUS],
        )
        df.rename(
            columns={
                f"max({mm_schemas.ResultData.RESULT_STATUS})": mm_schemas.ResultData.RESULT_STATUS
            },
            inplace=True,
        )
        if not df.empty:
            df.dropna(inplace=True)
        return df

    def get_metrics_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        start, end = self._get_start_end(start, end)
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.METRICS].super_table,
            start=start,
            end=end,
            columns=[
                mm_schemas.ApplicationEvent.APPLICATION_NAME,
                mm_schemas.MetricData.METRIC_NAME,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            filter_query=self._get_endpoint_filter(endpoint_id=endpoint_id),
            timestamp_column=mm_schemas.WriterEvent.END_INFER_TIME,
            group_by=[
                mm_schemas.WriterEvent.APPLICATION_NAME,
                mm_schemas.MetricData.METRIC_NAME,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            agg_funcs=["last"],
        )
        df.rename(
            columns={
                f"last({mm_schemas.ApplicationEvent.APPLICATION_NAME})": mm_schemas.ApplicationEvent.APPLICATION_NAME,
                f"last({mm_schemas.MetricData.METRIC_NAME})": mm_schemas.MetricData.METRIC_NAME,
                f"last({mm_schemas.EventFieldType.ENDPOINT_ID})": mm_schemas.EventFieldType.ENDPOINT_ID,
            },
            inplace=True,
        )
        if not df.empty:
            df.dropna(inplace=True)
        return df

    def get_results_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        start, end = self._get_start_end(start, end)
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.APP_RESULTS].super_table,
            start=start,
            end=end,
            columns=[
                mm_schemas.ApplicationEvent.APPLICATION_NAME,
                mm_schemas.ResultData.RESULT_NAME,
                mm_schemas.ResultData.RESULT_KIND,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            filter_query=self._get_endpoint_filter(endpoint_id=endpoint_id),
            timestamp_column=mm_schemas.WriterEvent.END_INFER_TIME,
            group_by=[
                mm_schemas.WriterEvent.APPLICATION_NAME,
                mm_schemas.ResultData.RESULT_NAME,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            agg_funcs=["last"],
        )
        df.rename(
            columns={
                f"last({mm_schemas.ApplicationEvent.APPLICATION_NAME})": mm_schemas.ApplicationEvent.APPLICATION_NAME,
                f"last({mm_schemas.ResultData.RESULT_NAME})": mm_schemas.ResultData.RESULT_NAME,
                f"last({mm_schemas.ResultData.RESULT_KIND})": mm_schemas.ResultData.RESULT_KIND,
                f"last({mm_schemas.EventFieldType.ENDPOINT_ID})": mm_schemas.EventFieldType.ENDPOINT_ID,
            },
            inplace=True,
        )
        if not df.empty:
            df.dropna(inplace=True)
        return df

    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> pd.DataFrame:
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_ids)
        filter_query += f"AND {mm_schemas.EventFieldType.ERROR_TYPE} = '{mm_schemas.EventFieldType.INFER_ERROR}'"
        start, end = self._get_start_end(start, end)
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.ERRORS].super_table,
            start=start,
            end=end,
            columns=[
                mm_schemas.EventFieldType.MODEL_ERROR,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            agg_funcs=["count"],
            filter_query=filter_query,
            group_by=mm_schemas.EventFieldType.ENDPOINT_ID,
            preform_agg_columns=[mm_schemas.EventFieldType.MODEL_ERROR],
        )
        df.rename(
            columns={f"count({mm_schemas.EventFieldType.MODEL_ERROR})": "error_count"},
            inplace=True,
        )
        if not df.empty:
            df.dropna(inplace=True)
        return df

    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> pd.DataFrame:
        endpoint_ids = (
            endpoint_ids if isinstance(endpoint_ids, list) else [endpoint_ids]
        )
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._get_start_end(start, end)
        df = self._get_records(
            table=self.tables[mm_schemas.TDEngineSuperTables.PREDICTIONS].super_table,
            start=start,
            end=end,
            columns=[
                mm_schemas.EventFieldType.LATENCY,
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            agg_funcs=["avg"],
            filter_query=f"endpoint_id IN({str(endpoint_ids)[1:-1]})",
            group_by=mm_schemas.EventFieldType.ENDPOINT_ID,
            preform_agg_columns=[mm_schemas.EventFieldType.LATENCY],
        )
        df.rename(
            columns={f"avg({mm_schemas.EventFieldType.LATENCY})": "avg_latency"},
            inplace=True,
        )
        if not df.empty:
            df.dropna(inplace=True)
        return df

    async def add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        project: str,
        run_in_threadpool: Callable,
        metric_list: Optional[list[str]] = None,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Add basic metrics to the model endpoint object.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will
                                        be filled with the relevant basic metrics.
        :param project:                The name of the project.
        :param run_in_threadpool:      A function that runs another function in a thread pool.
        :param metric_list:            List of metrics to include from the time series DB. Defaults to all metrics.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """

        uids = [mep.metadata.uid for mep in model_endpoint_objects]

        metric_name_to_function = {
            "error_count": self.get_error_count,
            "last_request": self.get_last_request,
            "avg_latency": self.get_avg_latency,
            "result_status": self.get_drift_status,
        }
        if metric_list is not None:
            for metric_name in list(metric_name_to_function):
                if metric_name not in metric_list:
                    del metric_name_to_function[metric_name]

        metric_name_to_df = {
            metric_name: function(endpoint_ids=uids)
            for metric_name, function in metric_name_to_function.items()
        }

        def add_metrics(
            mep: mlrun.common.schemas.ModelEndpoint,
            df_dictionary: dict[str, pd.DataFrame],
        ):
            for metric in df_dictionary.keys():
                df = df_dictionary.get(metric, pd.DataFrame())
                if not df.empty:
                    line = df[df["endpoint_id"] == mep.metadata.uid]
                    if not line.empty and metric in line:
                        value = line[metric].item()
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        setattr(mep.status, metric, value)

            return mep

        return list(
            map(
                lambda mep: add_metrics(
                    mep=mep,
                    df_dictionary=metric_name_to_df,
                ),
                model_endpoint_objects,
            )
        )

    # Note: this function serves as a reference for checking the TSDB for the existence of a metric.
    #
    # def read_prediction_metric_for_endpoint_if_exists(
    #     self, endpoint_id: str
    # ) -> Optional[mm_schemas.ModelEndpointMonitoringMetric]:
    #     """
    #     Read the "invocations" metric for the provided model endpoint, and return the metric object
    #     if it exists.
    #
    #     :param endpoint_id: The model endpoint identifier.
    #     :return:            `None` if the invocations metric does not exist, otherwise return the
    #                         corresponding metric object.
    #     """
    #     # Read just one record, because we just want to check if there is any data for this endpoint_id
    #     predictions = self.read_predictions(
    #         endpoint_id=endpoint_id,
    #         start=datetime.min,
    #         end=mlrun.utils.now_date(),
    #         limit=1,
    #     )
    #     if predictions:
    #         return get_invocations_metric(self.project)
