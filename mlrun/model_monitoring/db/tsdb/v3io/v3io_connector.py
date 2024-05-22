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
import datetime
import typing

import pandas as pd
import v3io_frames.client
import v3io_frames.errors
from v3io.dataplane import Client as V3IOClient
from v3io_frames.frames_pb2 import IGNORE

import mlrun.common.model_monitoring
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.feature_store.steps
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.utils import logger

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"


class V3IOTSDBConnector(TSDBConnector):
    """
    Handles the TSDB operations when the TSDB connector is of type V3IO. To manage these operations we use V3IO Frames
    Client that provides API for executing commands on the V3IO TSDB table.
    """

    type: str = mm_schemas.TSDBTarget.V3IO_TSDB

    def __init__(
        self,
        project: str,
        access_key: typing.Optional[str] = None,
        container: str = "users",
        v3io_framesd: typing.Optional[str] = None,
        create_table: bool = False,
    ):
        super().__init__(project=project)
        self.access_key = access_key or mlrun.mlconf.get_v3io_access_key()

        self.container = container

        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self._frames_client: v3io_frames.client.ClientBase = (
            self._get_v3io_frames_client(self.container)
        )
        self._v3io_client: V3IOClient = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api,
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
                if_exists=IGNORE,
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
            columns=["latency"],
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

        event[mm_schemas.WriterEvent.END_INFER_TIME] = datetime.datetime.fromisoformat(
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
        except v3io_frames.errors.Error as err:
            logger.exception(
                "Could not write drift measures to TSDB",
                err=err,
                table=table,
                event=event,
            )
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed to write application result to TSDB: {err}"
            )

    def delete_tsdb_resources(self, table: typing.Optional[str] = None):
        if table:
            # Delete a specific table
            tables = [table]
        else:
            # Delete all tables
            tables = mm_schemas.V3IOTSDBTables.list()
        for table in tables:
            try:
                self._frames_client.delete(
                    backend=mlrun.common.schemas.model_monitoring.TimeSeriesConnector.TSDB,
                    table=table,
                )
            except v3io_frames.errors.DeleteError as e:
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
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str,
        end: str,
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
            data = self.get_records(
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

        except v3io_frames.errors.Error as err:
            logger.warn("Failed to read tsdb", err=err, endpoint=endpoint_id)

        return metrics_mapping

    def get_records(
        self,
        table: str,
        start: str,
        end: str,
        columns: typing.Optional[list[str]] = None,
        filter_query: str = "",
    ) -> pd.DataFrame:
        """
         Getting records from V3IO TSDB data collection.
        :param table:            Path to the collection to query.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and
                                 `'s'` = seconds), or 0 for the earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and
                                 `'s'` = seconds), or 0 for the earliest time.
        :param columns:          Columns to include in the result.
        :param filter_query:     V3IO filter expression. The expected filter expression includes different conditions,
                                 divided by ' AND '.
        :return: DataFrame with the provided attributes from the data collection.
        :raise:  MLRunNotFoundError if the provided table wasn't found.
        """
        if table not in self.tables:
            raise mlrun.errors.MLRunNotFoundError(
                f"Table '{table}' does not exist in the tables list of the TSDB connector."
                f"Available tables: {list(self.tables.keys())}"
            )
        return self._frames_client.read(
            backend=mlrun.common.schemas.model_monitoring.TimeSeriesConnector.TSDB,
            table=self.tables[table],
            columns=columns,
            filter=filter_query,
            start=start,
            end=end,
        )

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
