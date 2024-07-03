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

from datetime import datetime
from io import StringIO
from typing import Literal, Optional, Union

import pandas as pd
import v3io_frames
import v3io_frames.client

import mlrun.common.model_monitoring
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.feature_store.steps
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.helpers import get_invocations_fqn
from mlrun.utils import logger

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"
_CONTAINER = "users"


def _is_no_schema_error(exc: v3io_frames.ReadError) -> bool:
    """
    In case of a nonexistent TSDB table - a `v3io_frames.ReadError` error is raised.
    Check if the error message contains the relevant string to verify the cause.
    """
    msg = str(exc)
    # https://github.com/v3io/v3io-tsdb/blob/v0.14.1/pkg/tsdb/v3iotsdb.go#L205
    # https://github.com/v3io/v3io-tsdb/blob/v0.14.1/pkg/partmgr/partmgr.go#L238
    return "No TSDB schema file found" in msg or "Failed to read schema at path" in msg


class V3IOTSDBConnector(TSDBConnector):
    """
    Handles the TSDB operations when the TSDB connector is of type V3IO. To manage these operations we use V3IO Frames
    Client that provides API for executing commands on the V3IO TSDB table.
    """

    type: str = mm_schemas.TSDBTarget.V3IO_TSDB

    def __init__(
        self,
        project: str,
        container: str = _CONTAINER,
        v3io_framesd: Optional[str] = None,
        create_table: bool = False,
    ) -> None:
        super().__init__(project=project)

        self.container = container

        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self._frames_client: v3io_frames.client.ClientBase = (
            self._get_v3io_frames_client(self.container)
        )

        self._init_tables_path()

        if create_table:
            self.create_tables()

    def _init_tables_path(self):
        self.tables = {}

        events_table_full_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mm_schemas.FileTargetKind.EVENTS,
        )
        (
            _,
            _,
            events_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            events_table_full_path
        )
        self.tables[mm_schemas.V3IOTSDBTables.EVENTS] = events_path

        monitoring_application_full_path = (
            mlrun.mlconf.get_model_monitoring_file_target_path(
                project=self.project,
                kind=mm_schemas.FileTargetKind.MONITORING_APPLICATION,
            )
        )
        (
            _,
            _,
            monitoring_application_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            monitoring_application_full_path
        )
        self.tables[mm_schemas.V3IOTSDBTables.APP_RESULTS] = (
            monitoring_application_path + mm_schemas.V3IOTSDBTables.APP_RESULTS
        )
        self.tables[mm_schemas.V3IOTSDBTables.METRICS] = (
            monitoring_application_path + mm_schemas.V3IOTSDBTables.METRICS
        )

        monitoring_predictions_full_path = (
            mlrun.mlconf.get_model_monitoring_file_target_path(
                project=self.project,
                kind=mm_schemas.FileTargetKind.PREDICTIONS,
            )
        )
        (
            _,
            _,
            monitoring_predictions_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            monitoring_predictions_full_path
        )
        self.tables[mm_schemas.FileTargetKind.PREDICTIONS] = monitoring_predictions_path

    def create_tables(self) -> None:
        """
        Create the tables using the TSDB connector. The tables are being created in the V3IO TSDB and include:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a single numeric metric.
        Note that the predictions table is automatically created by the model monitoring stream pod.
        """
        application_tables = [
            mm_schemas.V3IOTSDBTables.APP_RESULTS,
            mm_schemas.V3IOTSDBTables.METRICS,
        ]
        for table_name in application_tables:
            logger.info("Creating table in V3IO TSDB", table_name=table_name)
            table = self.tables[table_name]
            self._frames_client.create(
                backend=_TSDB_BE,
                table=table,
                if_exists=v3io_frames.IGNORE,
                rate=_TSDB_RATE,
            )

    def apply_monitoring_stream_steps(
        self,
        graph,
        tsdb_batching_max_events: int = 10,
        tsdb_batching_timeout_secs: int = 300,
    ):
        """
        Apply TSDB steps on the provided monitoring graph. Throughout these steps, the graph stores live data of
        different key metric dictionaries.This data is being used by the monitoring dashboards in
        grafana. Results can be found under  v3io:///users/pipelines/project-name/model-endpoints/events/.
        In that case, we generate 3 different key  metric dictionaries:
        - base_metrics (average latency and predictions over time)
        - endpoint_features (Prediction and feature names and values)
        - custom_metrics (user-defined metrics)
        """

        # Write latency per prediction, labeled by endpoint ID only
        graph.add_step(
            "storey.TSDBTarget",
            name="tsdb_predictions",
            after="MapFeatureNames",
            path=f"{self.container}/{self.tables[mm_schemas.FileTargetKind.PREDICTIONS]}",
            rate="1/s",
            time_col=mm_schemas.EventFieldType.TIMESTAMP,
            container=self.container,
            v3io_frames=self.v3io_framesd,
            columns=[mm_schemas.EventFieldType.LATENCY],
            index_cols=[
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            aggr="count,avg",
            aggr_granularity="1m",
            max_events=tsdb_batching_max_events,
            flush_after_seconds=tsdb_batching_timeout_secs,
            key=mm_schemas.EventFieldType.ENDPOINT_ID,
        )

        # Before writing data to TSDB, create dictionary of 2-3 dictionaries that contains
        # stats and details about the events

        def apply_process_before_tsdb():
            graph.add_step(
                "mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps.ProcessBeforeTSDB",
                name="ProcessBeforeTSDB",
                after="sample",
            )

        apply_process_before_tsdb()

        # Unpacked keys from each dictionary and write to TSDB target
        def apply_filter_and_unpacked_keys(name, keys):
            graph.add_step(
                "mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps.FilterAndUnpackKeys",
                name=name,
                after="ProcessBeforeTSDB",
                keys=[keys],
            )

        def apply_tsdb_target(name, after):
            graph.add_step(
                "storey.TSDBTarget",
                name=name,
                after=after,
                path=f"{self.container}/{self.tables[mm_schemas.V3IOTSDBTables.EVENTS]}",
                rate="10/m",
                time_col=mm_schemas.EventFieldType.TIMESTAMP,
                container=self.container,
                v3io_frames=self.v3io_framesd,
                infer_columns_from_data=True,
                index_cols=[
                    mm_schemas.EventFieldType.ENDPOINT_ID,
                    mm_schemas.EventFieldType.RECORD_TYPE,
                    mm_schemas.EventFieldType.ENDPOINT_TYPE,
                ],
                max_events=tsdb_batching_max_events,
                flush_after_seconds=tsdb_batching_timeout_secs,
                key=mm_schemas.EventFieldType.ENDPOINT_ID,
            )

        # unpacked base_metrics dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys1",
            keys=mm_schemas.EventKeyMetrics.BASE_METRICS,
        )
        apply_tsdb_target(name="tsdb1", after="FilterAndUnpackKeys1")

        # unpacked endpoint_features dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys2",
            keys=mm_schemas.EventKeyMetrics.ENDPOINT_FEATURES,
        )
        apply_tsdb_target(name="tsdb2", after="FilterAndUnpackKeys2")

        # unpacked custom_metrics dictionary. In addition, use storey.Filter remove none values
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys3",
            keys=mm_schemas.EventKeyMetrics.CUSTOM_METRICS,
        )

        def apply_storey_filter():
            graph.add_step(
                "storey.Filter",
                "FilterNotNone",
                after="FilterAndUnpackKeys3",
                _fn="(event is not None)",
            )

        apply_storey_filter()
        apply_tsdb_target(name="tsdb3", after="FilterNotNone")

    def write_application_event(
        self,
        event: dict,
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """Write a single result or metric to TSDB"""

        event[mm_schemas.WriterEvent.END_INFER_TIME] = datetime.fromisoformat(
            event[mm_schemas.WriterEvent.END_INFER_TIME]
        )
        index_cols_base = [
            mm_schemas.WriterEvent.END_INFER_TIME,
            mm_schemas.WriterEvent.ENDPOINT_ID,
            mm_schemas.WriterEvent.APPLICATION_NAME,
        ]

        if kind == mm_schemas.WriterEventKind.METRIC:
            table = self.tables[mm_schemas.V3IOTSDBTables.METRICS]
            index_cols = index_cols_base + [mm_schemas.MetricData.METRIC_NAME]
        elif kind == mm_schemas.WriterEventKind.RESULT:
            table = self.tables[mm_schemas.V3IOTSDBTables.APP_RESULTS]
            index_cols = index_cols_base + [mm_schemas.ResultData.RESULT_NAME]
            del event[mm_schemas.ResultData.RESULT_EXTRA_DATA]
        else:
            raise ValueError(f"Invalid {kind = }")

        try:
            self._frames_client.write(
                backend=_TSDB_BE,
                table=table,
                dfs=pd.DataFrame.from_records([event]),
                index_cols=index_cols,
            )
            logger.info("Updated V3IO TSDB successfully", table=table)
        except v3io_frames.Error as err:
            logger.exception(
                "Could not write drift measures to TSDB",
                err=err,
                table=table,
                event=event,
            )
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to write application result to TSDB: {err}"
            )

    def delete_tsdb_resources(self, table: Optional[str] = None):
        if table:
            # Delete a specific table
            tables = [table]
        else:
            # Delete all tables
            tables = mm_schemas.V3IOTSDBTables.list()
        for table_to_delete in tables:
            try:
                self._frames_client.delete(backend=_TSDB_BE, table=table_to_delete)
            except v3io_frames.DeleteError as e:
                logger.warning(
                    f"Failed to delete TSDB table '{table}'",
                    err=mlrun.errors.err_to_str(e),
                )

        # Final cleanup of tsdb path
        tsdb_path = self._get_v3io_source_directory()
        tsdb_path.replace("://u", ":///u")
        store, _, _ = mlrun.store_manager.get_or_create_store(tsdb_path)
        store.rm(tsdb_path, recursive=True)

    def get_model_endpoint_real_time_metrics(
        self, endpoint_id: str, metrics: list[str], start: str, end: str
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Getting real time metrics from the TSDB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user. Note that these
        metrics are being calculated by the model monitoring stream pod.
        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of real-time metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and
                                 `'s'` = seconds), or 0 for the earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days, and
                                 `'s'` = seconds), or 0 for the earliest time.
        :return: A dictionary of metrics in which the key is a metric name and the value is a list of tuples that
                 includes timestamps and the values.
        """

        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Metric names must be provided"
            )

        metrics_mapping = {}

        try:
            data = self._get_records(
                table=mm_schemas.V3IOTSDBTables.EVENTS,
                columns=["endpoint_id", *metrics],
                filter_query=f"endpoint_id=='{endpoint_id}'",
                start=start,
                end=end,
            )

            # Fill the metrics mapping dictionary with the metric name and values
            data_dict = data.to_dict()
            for metric in metrics:
                metric_data = data_dict.get(metric)
                if metric_data is None:
                    continue

                values = [
                    (str(timestamp), value) for timestamp, value in metric_data.items()
                ]
                metrics_mapping[metric] = values

        except v3io_frames.Error as err:
            logger.warn("Failed to read tsdb", err=err, endpoint=endpoint_id)

        return metrics_mapping

    def _get_records(
        self,
        table: str,
        start: Union[datetime, str],
        end: Union[datetime, str],
        columns: Optional[list[str]] = None,
        filter_query: str = "",
        interval: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        sliding_window_step: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
         Getting records from V3IO TSDB data collection.
        :param table:                 Path to the collection to query.
        :param start:                 The start time of the metrics. Can be represented by a string containing an RFC
                                      3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                      `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and
                                      `'s'` = seconds), or 0 for the earliest time.
        :param end:                   The end time of the metrics. Can be represented by a string containing an RFC
                                      3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                      `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and
                                      `'s'` = seconds), or 0 for the earliest time.
        :param columns:               Columns to include in the result.
        :param filter_query:          V3IO filter expression. The expected filter expression includes different
                                      conditions, divided by ' AND '.
        :param interval:              The interval to aggregate the data by. Note that if interval is provided,
                                      agg_funcs must bg provided as well. Provided as a string in the format of '1m',
                                      '1h', etc.
        :param agg_funcs:             The aggregation functions to apply on the columns. Note that if `agg_funcs` is
                                      provided, `interval` must bg provided as well. Provided as a list of strings in
                                      the format of ['sum', 'avg', 'count', ...].
        :param sliding_window_step:   The time step for which the time window moves forward. Note that if
                                      `sliding_window_step` is provided, interval must be provided as well. Provided
                                      as a string in the format of '1m', '1h', etc.
        :param kwargs:                Additional keyword arguments passed to the read method of frames client.
        :return: DataFrame with the provided attributes from the data collection.
        :raise:  MLRunNotFoundError if the provided table wasn't found.
        """
        if table not in self.tables:
            raise mlrun.errors.MLRunNotFoundError(
                f"Table '{table}' does not exist in the tables list of the TSDB connector. "
                f"Available tables: {list(self.tables.keys())}"
            )

        # Frames client expects the aggregators to be a comma-separated string
        aggregators = ",".join(agg_funcs) if agg_funcs else None
        table_path = self.tables[table]
        try:
            df = self._frames_client.read(
                backend=_TSDB_BE,
                table=table_path,
                start=start,
                end=end,
                columns=columns,
                filter=filter_query,
                aggregation_window=interval,
                aggregators=aggregators,
                step=sliding_window_step,
                **kwargs,
            )
        except v3io_frames.ReadError as err:
            if _is_no_schema_error(err):
                return pd.DataFrame()
            else:
                raise err

        return df

    def _get_v3io_source_directory(self) -> str:
        """
        Get the V3IO source directory for the current project. Usually the source directory will
        be under 'v3io:///users/pipelines/<project>'

        :return: The V3IO source directory for the current project.
        """
        events_table_full_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mm_schemas.FileTargetKind.EVENTS,
        )

        # Generate the main directory with the V3IO resources
        source_directory = (
            mlrun.common.model_monitoring.helpers.parse_model_endpoint_project_prefix(
                events_table_full_path, self.project
            )
        )

        return source_directory

    @staticmethod
    def _get_v3io_frames_client(v3io_container: str) -> v3io_frames.client.ClientBase:
        return mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=v3io_container,
        )

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: Literal["metrics", "results"] = "results",
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
        """
        Read metrics OR results from the TSDB and return as a list.
        Note: the type must match the actual metrics in the `metrics` parameter.
        If the type is "results", pass only results in the `metrics` parameter.
        """

        if type == "metrics":
            table_path = self.tables[mm_schemas.V3IOTSDBTables.METRICS]
            name = mm_schemas.MetricData.METRIC_NAME
            df_handler = self.df_to_metrics_values
        elif type == "results":
            table_path = self.tables[mm_schemas.V3IOTSDBTables.APP_RESULTS]
            name = mm_schemas.ResultData.RESULT_NAME
            df_handler = self.df_to_results_values
        else:
            raise ValueError(f"Invalid {type = }")

        query = self._get_sql_query(
            endpoint_id=endpoint_id,
            metric_and_app_names=[(metric.app, metric.name) for metric in metrics],
            table_path=table_path,
            name=name,
        )

        logger.debug("Querying V3IO TSDB", query=query)

        df: pd.DataFrame = self._frames_client.read(
            backend=_TSDB_BE,
            start=start,
            end=end,
            query=query,  # the filter argument does not work for this complex condition
        )

        logger.debug(
            "Converting a DataFrame to a list of metrics or results values",
            table=table_path,
            project=self.project,
            endpoint_id=endpoint_id,
            is_empty=df.empty,
        )

        return df_handler(df=df, metrics=metrics, project=self.project)

    @staticmethod
    def _get_sql_query(
        *,
        endpoint_id: str,
        table_path: str,
        name: str = mm_schemas.ResultData.RESULT_NAME,
        metric_and_app_names: Optional[list[tuple[str, str]]] = None,
        columns: Optional[list[str]] = None,
    ) -> str:
        """Get the SQL query for the results/metrics table"""
        if columns:
            selection = ",".join(columns)
        else:
            selection = "*"

        with StringIO() as query:
            query.write(
                f"SELECT {selection} FROM '{table_path}' "
                f"WHERE {mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_id}'"
            )
            if metric_and_app_names:
                query.write(" AND (")

                for i, (app_name, result_name) in enumerate(metric_and_app_names):
                    sub_cond = (
                        f"({mm_schemas.WriterEvent.APPLICATION_NAME}='{app_name}' "
                        f"AND {name}='{result_name}')"
                    )
                    if i != 0:  # not first sub condition
                        query.write(" OR ")
                    query.write(sub_cond)

                query.write(")")

            query.write(";")
            return query.getvalue()

    def read_predictions(
        self,
        *,
        endpoint_id: str,
        start: Union[datetime, str],
        end: Union[datetime, str],
        aggregation_window: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
    ) -> Union[
        mm_schemas.ModelEndpointMonitoringMetricNoData,
        mm_schemas.ModelEndpointMonitoringMetricValues,
    ]:
        if (agg_funcs and not aggregation_window) or (
            aggregation_window and not agg_funcs
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "both or neither of `aggregation_window` and `agg_funcs` must be provided"
            )
        df = self._get_records(
            table=mm_schemas.FileTargetKind.PREDICTIONS,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.LATENCY],
            filter_query=f"endpoint_id=='{endpoint_id}'",
            interval=aggregation_window,
            agg_funcs=agg_funcs,
            sliding_window_step=aggregation_window,
        )

        full_name = get_invocations_fqn(self.project)

        if df.empty:
            return mm_schemas.ModelEndpointMonitoringMetricNoData(
                full_name=full_name,
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )

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

    # Note: this function serves as a reference for checking the TSDB for the existence of a metric.
    #
    # def read_prediction_metric_for_endpoint_if_exists(
    #     self, endpoint_id: str
    # ) -> Optional[mm_schemas.ModelEndpointMonitoringMetric]:
    #     """
    #     Read the count of the latency column in the predictions table for the given endpoint_id.
    #     We just want to check if there is any data for this endpoint_id.
    #     """
    #     query = self._get_sql_query(
    #         endpoint_id=endpoint_id,
    #         table_path=self.tables[mm_schemas.FileTargetKind.PREDICTIONS],
    #         columns=[f"count({mm_schemas.EventFieldType.LATENCY})"],
    #     )
    #     try:
    #         logger.debug("Checking TSDB", project=self.project, query=query)
    #         df: pd.DataFrame = self._frames_client.read(
    #             backend=_TSDB_BE, query=query, start="0", end="now"
    #         )
    #     except v3io_frames.ReadError as err:
    #         if _is_no_schema_error(err):
    #             logger.debug(
    #                 "No predictions yet", project=self.project, endpoint_id=endpoint_id
    #             )
    #             return
    #         else:
    #             raise
    #
    #     if not df.empty:
    #         return get_invocations_metric(self.project)
