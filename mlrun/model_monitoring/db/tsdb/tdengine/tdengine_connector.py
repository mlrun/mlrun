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

import typing
from datetime import datetime
from typing import Union

import pandas as pd
import taosws

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.tdengine.schemas as tdengine_schemas
import mlrun.model_monitoring.db.tsdb.tdengine.stream_graph_steps
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.helpers import get_invocations_fqn
from mlrun.utils import logger


class TDEngineConnector(TSDBConnector):
    """
    Handles the TSDB operations when the TSDB connector is of type TDEngine.
    """

    type: str = mm_schemas.TSDBTarget.TDEngine

    def __init__(
        self,
        project: str,
        database: str = tdengine_schemas._MODEL_MONITORING_DATABASE,
        **kwargs,
    ):
        super().__init__(project=project)
        if "connection_string" not in kwargs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "connection_string is a required parameter for TDEngineConnector."
            )
        self._tdengine_connection_string = kwargs.get("connection_string")
        self.database = database

        self._connection = None
        self._init_super_tables()

    @property
    def connection(self) -> taosws.Connection:
        if not self._connection:
            self._connection = self._create_connection()
        return self._connection

    def with_retry_on_closed_connection(self, fn, **kwargs):
        try:
            return fn(self.connection, **kwargs)
        except (taosws.QueryError, taosws.FetchError) as err:
            logger.warn(f"TDEngine error: {err}")
            if "Internal error:" in str(err):
                logger.info("Retrying TDEngine query with a new connection")
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None
                return fn(self.connection, **kwargs)
            else:
                raise err

    def _create_connection(self) -> taosws.Connection:
        """Establish a connection to the TSDB server."""
        logger.debug("Creating a new connection to TDEngine", project=self.project)
        conn = taosws.connect(self._tdengine_connection_string)
        try:
            conn.execute(f"CREATE DATABASE {self.database}")
        except taosws.QueryError:
            # Database already exists
            pass
        try:
            conn.execute(f"USE {self.database}")
        except taosws.QueryError as e:
            raise mlrun.errors.MLRunTSDBConnectionFailureError(
                f"Failed to use TDEngine database {self.database}, {mlrun.errors.err_to_str(e)}"
            )
        logger.debug("Connected to TDEngine", project=self.project)
        return conn

    def _init_super_tables(self):
        """Initialize the super tables for the TSDB."""
        self.tables = {
            mm_schemas.TDEngineSuperTables.APP_RESULTS: tdengine_schemas.AppResultTable(
                self.database
            ),
            mm_schemas.TDEngineSuperTables.METRICS: tdengine_schemas.Metrics(
                self.database
            ),
            mm_schemas.TDEngineSuperTables.PREDICTIONS: tdengine_schemas.Predictions(
                self.database
            ),
        }

    def create_tables(self):
        """Create TDEngine supertables."""
        for table in self.tables:
            create_table_query = self.tables[table]._create_super_table_query()
            self.with_retry_on_closed_connection(
                lambda conn: conn.execute(create_table_query)
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
            f"{self.project}_"
            f"{event[mm_schemas.WriterEvent.ENDPOINT_ID]}_"
            f"{event[mm_schemas.WriterEvent.APPLICATION_NAME]}_"
        )
        event[mm_schemas.EventFieldType.PROJECT] = self.project

        if kind == mm_schemas.WriterEventKind.RESULT:
            # Write a new result
            table = self.tables[mm_schemas.TDEngineSuperTables.APP_RESULTS]
            table_name = (
                f"{table_name}_{event[mm_schemas.ResultData.RESULT_NAME]}"
            ).replace("-", "_")
            event.pop(mm_schemas.ResultData.CURRENT_STATS, None)

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
        self.with_retry_on_closed_connection(
            lambda conn: conn.execute(create_table_sql)
        )

        insert_statement = self.with_retry_on_closed_connection(
            lambda conn: table._insert_subtable_stmt(
                conn, subtable=table_name, values=event
            )
        )
        insert_statement.add_batch()
        insert_statement.execute()

    @staticmethod
    def _convert_to_datetime(val: typing.Union[str, datetime]) -> datetime:
        return datetime.fromisoformat(val) if isinstance(val, str) else val

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
                supertable=mm_schemas.TDEngineSuperTables.PREDICTIONS,
                table_col=mm_schemas.EventFieldType.TABLE_COLUMN,
                time_col=mm_schemas.EventFieldType.TIME,
                database=self.database,
                columns=[
                    mm_schemas.EventFieldType.LATENCY,
                    mm_schemas.EventKeyMetrics.CUSTOM_METRICS,
                ],
                tag_cols=[
                    mm_schemas.EventFieldType.PROJECT,
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

    def handle_model_error(self, graph, **kwargs) -> None:
        pass

    def delete_tsdb_resources(self):
        """
        Delete all project resources in the TSDB connector, such as model endpoints data and drift results.
        """
        logger.debug(
            "Deleting all project resources using the TDEngine connector",
            project=self.project,
        )
        for table in self.tables:
            get_subtable_names_query = self.tables[table]._get_subtables_query(
                values={mm_schemas.EventFieldType.PROJECT: self.project}
            )
            subtables = self.with_retry_on_closed_connection(
                lambda conn: conn.query(get_subtable_names_query)
            )
            for subtable in subtables:
                drop_query = self.tables[table]._drop_subtable_query(
                    subtable=subtable[0]
                )
                self.connection.execute(drop_query)
        logger.debug(
            "Deleted all project resources using the TDEngine connector",
            project=self.project,
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
        columns: typing.Optional[list[str]] = None,
        filter_query: typing.Optional[str] = None,
        interval: typing.Optional[str] = None,
        agg_funcs: typing.Optional[list] = None,
        limit: typing.Optional[int] = None,
        sliding_window_step: typing.Optional[str] = None,
        timestamp_column: str = mm_schemas.EventFieldType.TIME,
        group_by: typing.Optional[Union[list[str], str]] = None,
        preform_agg_columns: typing.Optional[list] = None,
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

        :return: DataFrame with the provided attributes from the data collection.
        :raise:  MLRunInvalidArgumentError if query the provided table failed.
        """

        project_condition = f"project = '{self.project}'"
        filter_query = (
            f"({filter_query}) AND ({project_condition})"
            if filter_query
            else project_condition
        )

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
            preform_agg_funcs_columns=preform_agg_columns
        )
        logger.debug("Querying TDEngine", query=full_query)
        try:
            query_result = self.with_retry_on_closed_connection(
                lambda conn: conn.query(full_query)
            )
        except taosws.QueryError as e:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Failed to query table {table} in database {self.database}, {str(e)}"
            )

        df_columns = [field.name() for field in query_result.fields]
        return pd.DataFrame(query_result, columns=df_columns)

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: typing.Literal["metrics", "results"],
    ) -> typing.Union[
        list[
            typing.Union[
                mm_schemas.ModelEndpointMonitoringResultValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
        list[
            typing.Union[
                mm_schemas.ModelEndpointMonitoringMetricValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
    ]:
        timestamp_column = mm_schemas.WriterEvent.END_INFER_TIME
        columns = [timestamp_column, mm_schemas.WriterEvent.APPLICATION_NAME]
        if type == "metrics":
            table = mm_schemas.TDEngineSuperTables.METRICS
            name = mm_schemas.MetricData.METRIC_NAME
            columns += [name, mm_schemas.MetricData.METRIC_VALUE]
            df_handler = self.df_to_metrics_values
        elif type == "results":
            table = mm_schemas.TDEngineSuperTables.APP_RESULTS
            name = mm_schemas.ResultData.RESULT_NAME
            columns += [
                name,
                mm_schemas.ResultData.RESULT_VALUE,
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.ResultData.RESULT_KIND,
            ]
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

        return df_handler(df=df, metrics=metrics, project=self.project)

    def read_predictions(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        aggregation_window: typing.Optional[str] = None,
        agg_funcs: typing.Optional[list] = None,
        limit: typing.Optional[int] = None,
    ) -> typing.Union[
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
            table=mm_schemas.TDEngineSuperTables.PREDICTIONS,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.LATENCY],
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

        latency_column = (
            f"{agg_funcs[0]}({mm_schemas.EventFieldType.LATENCY})"
            if agg_funcs
            else mm_schemas.EventFieldType.LATENCY
        )

        return mm_schemas.ModelEndpointMonitoringMetricValues(
            full_name=full_name,
            values=list(
                zip(
                    df.index,
                    df[latency_column],
                )
            ),  # pyright: ignore[reportArgumentType]
        )

    def get_last_request(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Union[datetime, str] = "0",
        end: Union[datetime, str] = "now",
    ) -> pd.DataFrame:
        pass

    def get_drift_status(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Union[datetime, str] = "now-24h",
        end: Union[datetime, str] = "now",
    ) -> pd.DataFrame:
        endpoint_ids = (
            endpoint_ids if isinstance(endpoint_ids, list) else [endpoint_ids]
        )
        df = self._get_records(
            table=mm_schemas.TDEngineSuperTables.APP_RESULTS,
            start=start,
            end=end,
            columns=[mm_schemas.ResultData.RESULT_STATUS, mm_schemas.SchedulingKeys.ENDPOINT_ID],
            filter_query=f"endpoint_id IN({str(endpoint_ids)[1:-1]})",
            timestamp_column=mm_schemas.WriterEvent.END_INFER_TIME,
            agg_funcs=["max"],
            group_by=mm_schemas.SchedulingKeys.ENDPOINT_ID,
            preform_agg_columns=[mm_schemas.ResultData.RESULT_STATUS]
        )
        if df.empty:
            df.dropna(inplace=True)
            df.rename(
                columns={
                    f"max({mm_schemas.ResultData.RESULT_STATUS})": mm_schemas.ResultData.RESULT_STATUS
                },
                inplace=True,
            )
        return df

    def get_metrics_metadata(
        self,
        endpoint_id: str,
        start: Union[datetime, str] = "0",
        end: Union[datetime, str] = "now",
    ) -> pd.DataFrame:
        df = self._get_records(
            table=mm_schemas.TDEngineSuperTables.METRICS,
            start=start,
            end=end,
            columns=[mm_schemas.ApplicationEvent.APPLICATION_NAME, mm_schemas.MetricData.METRIC_NAME,
                     mm_schemas.SchedulingKeys.ENDPOINT_ID],
            filter_query=f"endpoint_id='{endpoint_id}'",
            timestamp_column=mm_schemas.WriterEvent.END_INFER_TIME,
            group_by=[mm_schemas.WriterEvent.APPLICATION_NAME, mm_schemas.MetricData.METRIC_NAME],
            agg_funcs=["last"],
        )
        if not df.empty:
            df.dropna(inplace=True)
            df.rename(
                columns={
                    f"last({mm_schemas.ApplicationEvent.APPLICATION_NAME})": mm_schemas.ApplicationEvent.APPLICATION_NAME,
                    f"last({mm_schemas.MetricData.METRIC_NAME})": mm_schemas.MetricData.METRIC_NAME,
                    f"last({mm_schemas.SchedulingKeys.ENDPOINT_ID})": mm_schemas.SchedulingKeys.ENDPOINT_ID
                },
                inplace=True,
            )
        return df


    def get_results_metadata(
        self,
        endpoint_id: str,
        start: Union[datetime, str] = "0",
        end: Union[datetime, str] = "now",
    ) -> pd.DataFrame:
        df = self._get_records(
            table=mm_schemas.TDEngineSuperTables.APP_RESULTS,
            start=start,
            end=end,
            columns=[mm_schemas.ApplicationEvent.APPLICATION_NAME, mm_schemas.ResultData.RESULT_NAME
                ,mm_schemas.ResultData.RESULT_KIND, mm_schemas.SchedulingKeys.ENDPOINT_ID],
            filter_query=f"endpoint_id='{endpoint_id}'",
            timestamp_column=mm_schemas.WriterEvent.END_INFER_TIME,
            group_by=[mm_schemas.WriterEvent.APPLICATION_NAME, mm_schemas.ResultData.RESULT_NAME],
            agg_funcs=["last"],
            )
        if not df.empty:
            df.dropna(inplace=True)
            df.rename(
                columns={
                    f"last({mm_schemas.ApplicationEvent.APPLICATION_NAME})": mm_schemas.ApplicationEvent.APPLICATION_NAME,
                    f"last({mm_schemas.ResultData.RESULT_NAME})": mm_schemas.ResultData.RESULT_NAME,
                    f"last({mm_schemas.ResultData.RESULT_KIND})": mm_schemas.ResultData.RESULT_KIND,
                    f"last({mm_schemas.SchedulingKeys.ENDPOINT_ID})": mm_schemas.SchedulingKeys.ENDPOINT_ID
                },
                inplace=True,
            )
        return df

    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Union[datetime, str] = "0",
        end: Union[datetime, str] = "now",
    ) -> pd.DataFrame:
        pass

    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Union[datetime, str] = "0",
        end: Union[datetime, str] = "now",
    ) -> pd.DataFrame:
        endpoint_ids = (
            endpoint_ids if isinstance(endpoint_ids, list) else [endpoint_ids]
        )
        df = self._get_records(
            table=mm_schemas.TDEngineSuperTables.PREDICTIONS,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.LATENCY, mm_schemas.SchedulingKeys.ENDPOINT_ID],
            agg_funcs=["avg"],
            filter_query=f"endpoint_id IN({str(endpoint_ids)[1:-1]})",
            group_by=mm_schemas.SchedulingKeys.ENDPOINT_ID,
            preform_agg_columns=[mm_schemas.EventFieldType.LATENCY]
        )
        if not df.empty:
            df.dropna(inplace=True)
            df.rename(
                columns={
                    f"avg({mm_schemas.EventFieldType.LATENCY})": "avg_latency"
                },
                inplace=True,
            )
        return df

    # Note: this function serves as a reference for checking the TSDB for the existence of a metric.
    #
    # def read_prediction_metric_for_endpoint_if_exists(
    #     self, endpoint_id: str
    # ) -> typing.Optional[mm_schemas.ModelEndpointMonitoringMetric]:
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
