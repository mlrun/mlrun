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
import datetime

import pandas as pd
import v3io_frames.client
import v3io_frames.errors
from v3io.dataplane import Client as V3IOClient
from v3io_frames.frames_pb2 import IGNORE

import mlrun.common.model_monitoring
import mlrun.common.schemas.model_monitoring as mm_constants
import mlrun.feature_store.steps
import mlrun.model_monitoring.db
import mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps
import mlrun.utils.v3io_clients
from mlrun.utils import logger

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"


class V3IOTSDBConnector(mlrun.model_monitoring.db.TSDBConnector):
    """
    Handles the TSDB operations when the TSDB connector is from type V3IO. To manage these operations we use V3IO Frames
    Client that provides API for executing commands on the V3IO TSDB table.
    """

    def __init__(
        self,
        project: str,
        access_key: str = None,
        container: str = "users",
        v3io_framesd: str = None,
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
            self.create_tsdb_application_tables()

    def _init_tables_path(self):
        self.tables = {}

        events_table_full_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mm_constants.FileTargetKind.EVENTS,
        )
        (
            _,
            _,
            events_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            events_table_full_path
        )
        self.tables[mm_constants.V3IOTSDBTables.EVENTS] = events_path

        monitoring_application_full_path = (
            mlrun.mlconf.get_model_monitoring_file_target_path(
                project=self.project,
                kind=mm_constants.FileTargetKind.MONITORING_APPLICATION,
            )
        )
        (
            _,
            _,
            monitoring_application_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            monitoring_application_full_path
        )
        self.tables[mm_constants.V3IOTSDBTables.APP_RESULTS] = (
            monitoring_application_path + mm_constants.V3IOTSDBTables.APP_RESULTS
        )
        self.tables[mm_constants.V3IOTSDBTables.METRICS] = (
            monitoring_application_path + mm_constants.V3IOTSDBTables.METRICS
        )

    def create_tsdb_application_tables(self):
        """
        Create the application tables using the TSDB connector. At the moment we support 2 types of application tables:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a single numeric metric.
        """
        application_tables = [
            mm_constants.V3IOTSDBTables.APP_RESULTS,
            mm_constants.V3IOTSDBTables.METRICS,
        ]
        for table in application_tables:
            logger.info("Creating table in V3IO TSDB", table=table)
            self._frames_client.create(
                backend=_TSDB_BE,
                table=self.tables[table],
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
        - custom_metrics (user-defined metrics
        """

        # Step 12 - Before writing data to TSDB, create dictionary of 2-3 dictionaries that contains
        # stats and details about the events

        def apply_process_before_tsdb():
            graph.add_step(
                "mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps.ProcessBeforeTSDB",
                name="ProcessBeforeTSDB",
                after="sample",
            )

        apply_process_before_tsdb()

        # Steps 13-19: - Unpacked keys from each dictionary and write to TSDB target
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
                path=self.tables[mm_constants.V3IOTSDBTables.EVENTS],
                rate="10/m",
                time_col=mm_constants.EventFieldType.TIMESTAMP,
                container=self.container,
                v3io_frames=self.v3io_framesd,
                infer_columns_from_data=True,
                index_cols=[
                    mm_constants.EventFieldType.ENDPOINT_ID,
                    mm_constants.EventFieldType.RECORD_TYPE,
                    mm_constants.EventFieldType.ENDPOINT_TYPE,
                ],
                max_events=tsdb_batching_max_events,
                flush_after_seconds=tsdb_batching_timeout_secs,
                key=mm_constants.EventFieldType.ENDPOINT_ID,
            )

        # Steps 13-14 - unpacked base_metrics dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys1",
            keys=mm_constants.EventKeyMetrics.BASE_METRICS,
        )
        apply_tsdb_target(name="tsdb1", after="FilterAndUnpackKeys1")

        # Steps 15-16 - unpacked endpoint_features dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys2",
            keys=mm_constants.EventKeyMetrics.ENDPOINT_FEATURES,
        )
        apply_tsdb_target(name="tsdb2", after="FilterAndUnpackKeys2")

        # Steps 17-19 - unpacked custom_metrics dictionary. In addition, use storey.Filter remove none values
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys3",
            keys=mm_constants.EventKeyMetrics.CUSTOM_METRICS,
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

    def write_application_event(self, event: dict):
        """
        Write a single application result event to TSDB.
        """
        event[mm_constants.WriterEvent.END_INFER_TIME] = (
            datetime.datetime.fromisoformat(
                event[mm_constants.WriterEvent.END_INFER_TIME]
            )
        )
        del event[mm_constants.WriterEvent.RESULT_EXTRA_DATA]
        try:
            self._frames_client.write(
                backend=_TSDB_BE,
                table=self.tables[mm_constants.V3IOTSDBTables.APP_RESULTS],
                dfs=pd.DataFrame.from_records([event]),
                index_cols=[
                    mm_constants.WriterEvent.END_INFER_TIME,
                    mm_constants.WriterEvent.ENDPOINT_ID,
                    mm_constants.WriterEvent.APPLICATION_NAME,
                    mm_constants.WriterEvent.RESULT_NAME,
                ],
            )
            logger.info(
                "Updated V3IO TSDB successfully",
                table=self.tables[mm_constants.V3IOTSDBTables.APP_RESULTS],
            )
        except v3io_frames.errors.Error as err:
            logger.warn(
                "Could not write drift measures to TSDB",
                err=err,
                table=self.tables[mm_constants.V3IOTSDBTables.APP_RESULTS],
                event=event,
            )

    def delete_tsdb_resources(self, table: str = None):
        if table:
            # Delete a specific table
            tables = [table]
        else:
            # Delete all tables
            tables = mm_constants.V3IOTSDBTables.list()
        for table in tables:
            try:
                self._frames_client.delete(
                    backend=mlrun.common.schemas.model_monitoring.TimeSeriesConnector.TSDB,
                    table=table,
                )
            except v3io_frames.errors.DeleteError as e:
                if "No TSDB schema file found" not in str(e):
                    logger.warning(
                        f"Failed to delete TSDB table '{table}'",
                        err=mlrun.errors.err_to_str(e),
                    )

    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str = "now-1h",
        end: str = "now",
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Getting metrics from the time series DB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user.
        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of real-time metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
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
                table=mm_constants.V3IOTSDBTables.EVENTS,
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
        columns: list[str] = None,
        filter_query: str = "",
        start: str = "now-1h",
        end: str = "now",
    ) -> pd.DataFrame:
        """
         Getting records from V3IO TSDB data collection.
        :param table:            Path to the collection to query.
        :param columns:          Columns to include in the result.
        :param filter_query:     V3IO filter expression. The expected filter expression includes different conditions,
                                 divided by ' AND '.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :return: DataFrame with the provided attributes from the data collection.
        """
        if table not in self.tables:
            raise mlrun.errors.MLRunInvalidArgumentError(
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

    @staticmethod
    def _get_v3io_frames_client(v3io_container: str) -> v3io_frames.client.ClientBase:
        return mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=v3io_container,
        )
