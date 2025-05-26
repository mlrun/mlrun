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
import math
from datetime import datetime, timedelta
from io import StringIO
from typing import Callable, Literal, Optional, Union

import pandas as pd
import v3io_frames
import v3io_frames.client

import mlrun.common.model_monitoring
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.feature_store.steps
import mlrun.utils.v3io_clients
from mlrun.common.schemas import EventFieldType
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.helpers import get_invocations_fqn
from mlrun.utils import logger

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"
_CONTAINER = "users"

V3IO_FRAMESD_MEPS_LIMIT = (
    200  # Maximum number of model endpoints per single request when using V3IO Frames
)
V3IO_CLIENT_MEPS_LIMIT = (
    150  # Maximum number of model endpoints per single request when using V3IO Client
)


def _is_no_schema_error(exc: v3io_frames.Error) -> bool:
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
        v3io_access_key: str = "",
        create_table: bool = False,
    ) -> None:
        super().__init__(project=project)

        self.container = container

        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self._v3io_access_key = v3io_access_key
        self._frames_client: Optional[v3io_frames.client.ClientBase] = None
        self._init_tables_path()
        self._create_table = create_table
        self._v3io_client = None

    @property
    def v3io_client(self):
        if not self._v3io_client:
            self._v3io_client = mlrun.utils.v3io_clients.get_v3io_client(
                endpoint=mlrun.mlconf.v3io_api, access_key=self._v3io_access_key
            )
        return self._v3io_client

    @property
    def frames_client(self) -> v3io_frames.client.ClientBase:
        if not self._frames_client:
            self._frames_client = self._get_v3io_frames_client(
                self.container, v3io_access_key=self._v3io_access_key
            )
            if self._create_table:
                self.create_tables()
        return self._frames_client

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

        errors_table_full_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mm_schemas.FileTargetKind.ERRORS,
        )
        (
            _,
            _,
            errors_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            errors_table_full_path
        )
        self.tables[mm_schemas.V3IOTSDBTables.ERRORS] = errors_path

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
                kind=mm_schemas.V3IOTSDBTables.PREDICTIONS,
            )
        )
        (
            _,
            _,
            monitoring_predictions_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            monitoring_predictions_full_path
        )
        self.tables[mm_schemas.V3IOTSDBTables.PREDICTIONS] = monitoring_predictions_path

        # initialize kv table
        last_request_full_table_path = (
            mlrun.mlconf.get_model_monitoring_file_target_path(
                project=self.project,
                kind=mm_schemas.FileTargetKind.LAST_REQUEST,
            )
        )
        (
            _,
            _,
            self.last_request_table,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            last_request_full_table_path
        )

    def create_tables(self) -> None:
        """
        Create the tables using the TSDB connector. These are the tables that are stored in the V3IO TSDB:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a single numeric metric.
        - events: A statistics table that includes pre-aggregated metrics (such as average latency over the
        last 5 minutes) and data samples
        - predictions: a detailed prediction that includes latency, request timestamp, etc. This table also
        includes pre-aggregated operations such as count and average on 1 minute granularity.
        - errors: a detailed error that includes error desc, error type, etc.

        """

        default_configurations = {
            "backend": _TSDB_BE,
            "if_exists": v3io_frames.IGNORE,
            "rate": _TSDB_RATE,
        }

        for table_name in self.tables:
            default_configurations["table"] = self.tables[table_name]
            if table_name == mm_schemas.V3IOTSDBTables.PREDICTIONS:
                default_configurations["aggregates"] = "count,avg"
                default_configurations["aggregation_granularity"] = "1m"
            elif table_name == mm_schemas.V3IOTSDBTables.EVENTS:
                default_configurations["rate"] = "10/m"
            logger.info("Creating table in V3IO TSDB", table_name=table_name)
            self.frames_client.create(**default_configurations)

    def apply_monitoring_stream_steps(
        self,
        graph,
        tsdb_batching_max_events: int = 1000,
        tsdb_batching_timeout_secs: int = 30,
        sample_window: int = 10,
        aggregate_windows: Optional[list[str]] = None,
        aggregate_period: str = "1m",
        **kwarg,
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
        aggregate_windows = aggregate_windows or ["5m", "1h"]

        # Calculate number of predictions and average latency
        def apply_storey_aggregations():
            # Calculate number of predictions for each window (5 min and 1 hour by default)
            graph.add_step(
                class_name="storey.AggregateByKey",
                aggregates=[
                    {
                        "name": EventFieldType.LATENCY,
                        "column": EventFieldType.LATENCY,
                        "operations": ["count", "avg"],
                        "windows": aggregate_windows,
                        "period": aggregate_period,
                    }
                ],
                name=EventFieldType.LATENCY,
                after="FilterNOP",
                step_name="Aggregates",
                table=".",
                key_field=EventFieldType.ENDPOINT_ID,
            )
            # Calculate average latency time for each window (5 min and 1 hour by default)
            graph.add_step(
                class_name="storey.Rename",
                mapping={
                    "latency_count_5m": mm_schemas.EventLiveStats.PREDICTIONS_COUNT_5M,
                    "latency_count_1h": mm_schemas.EventLiveStats.PREDICTIONS_COUNT_1H,
                },
                name="Rename",
                after=EventFieldType.LATENCY,
            )

        apply_storey_aggregations()
        # Write latency per prediction, labeled by endpoint ID only
        graph.add_step(
            "storey.TSDBTarget",
            name="tsdb_predictions",
            after="FilterNOP",
            path=f"{self.container}/{self.tables[mm_schemas.V3IOTSDBTables.PREDICTIONS]}",
            time_col=mm_schemas.EventFieldType.TIMESTAMP,
            container=self.container,
            v3io_frames=self.v3io_framesd,
            columns=[
                mm_schemas.EventFieldType.LATENCY,
                mm_schemas.EventFieldType.LAST_REQUEST_TIMESTAMP,
                mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT,
                mm_schemas.EventFieldType.EFFECTIVE_SAMPLE_COUNT,
            ],
            index_cols=[
                mm_schemas.EventFieldType.ENDPOINT_ID,
            ],
            max_events=tsdb_batching_max_events,
            flush_after_seconds=tsdb_batching_timeout_secs,
            key=mm_schemas.EventFieldType.ENDPOINT_ID,
        )

        # Write last request timestamp to KV table
        graph.add_step(
            "storey.NoSqlTarget",
            name="KVLastRequest",
            after="tsdb_predictions",
            table=f"v3io:///users/{self.last_request_table}",
            columns=[EventFieldType.LAST_REQUEST_TIMESTAMP],
            index_cols=[EventFieldType.ENDPOINT_ID],
        )

        # Emits the event in window size of events based on sample_window size (10 by default)
        graph.add_step(
            "storey.steps.SampleWindow",
            name="sample",
            after="Rename",
            window_size=sample_window,
            key=EventFieldType.ENDPOINT_ID,
        )

        # Before writing data to TSDB, create dictionary of 2-3 dictionaries that contains
        # stats and details about the events

        graph.add_step(
            "mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps.ProcessBeforeTSDB",
            name="ProcessBeforeTSDB",
            after="sample",
        )

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

    def handle_model_error(
        self,
        graph,
        tsdb_batching_max_events: int = 1000,
        tsdb_batching_timeout_secs: int = 30,
        **kwargs,
    ) -> None:
        graph.add_step(
            "mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps.ErrorExtractor",
            name="error_extractor",
            after="ForwardError",
        )

        graph.add_step(
            "storey.TSDBTarget",
            name="tsdb_error",
            after="error_extractor",
            path=f"{self.container}/{self.tables[mm_schemas.FileTargetKind.ERRORS]}",
            time_col=mm_schemas.EventFieldType.TIMESTAMP,
            container=self.container,
            v3io_frames=self.v3io_framesd,
            columns=[
                mm_schemas.EventFieldType.MODEL_ERROR,
                mm_schemas.EventFieldType.ERROR_COUNT,
            ],
            index_cols=[
                mm_schemas.EventFieldType.ENDPOINT_ID,
                mm_schemas.EventFieldType.ERROR_TYPE,
            ],
            max_events=tsdb_batching_max_events,
            flush_after_seconds=tsdb_batching_timeout_secs,
            key=mm_schemas.EventFieldType.ENDPOINT_ID,
        )

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
        else:
            raise ValueError(f"Invalid {kind = }")

        try:
            self.frames_client.write(
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
            if table_to_delete in self.tables:
                try:
                    self.frames_client.delete(
                        backend=_TSDB_BE, table=self.tables[table_to_delete]
                    )
                except v3io_frames.DeleteError as e:
                    logger.warning(
                        f"Failed to delete TSDB table '{table_to_delete}'",
                        err=mlrun.errors.err_to_str(e),
                    )
            else:
                logger.warning(
                    f"Skipping deletion: table '{table_to_delete}' is not among the initialized tables.",
                    initialized_tables=list(self.tables.keys()),
                )

        # Final cleanup of tsdb path
        tsdb_path = self._get_v3io_source_directory()
        tsdb_path.replace("://u", ":///u")
        store, _, _ = mlrun.store_manager.get_or_create_store(tsdb_path)
        store.rm(tsdb_path, recursive=True)

    def delete_tsdb_records(
        self,
        endpoint_ids: list[str],
    ):
        logger.debug(
            "Deleting model endpoints resources using the V3IO TSDB connector",
            project=self.project,
            number_of_endpoints_to_delete=len(endpoint_ids),
        )
        tables = mm_schemas.V3IOTSDBTables.list()

        # Split the endpoint ids into chunks to avoid exceeding the v3io-engine filter-expression limit
        for i in range(0, len(endpoint_ids), V3IO_FRAMESD_MEPS_LIMIT):
            endpoint_id_chunk = endpoint_ids[i : i + V3IO_FRAMESD_MEPS_LIMIT]
            filter_query = f"endpoint_id IN({str(endpoint_id_chunk)[1:-1]}) "
            for table in tables:
                try:
                    self.frames_client.delete(
                        backend=_TSDB_BE,
                        table=self.tables[table],
                        filter=filter_query,
                        start="0",
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete TSDB records for the provided endpoints from table '{table}'",
                        error=mlrun.errors.err_to_str(e),
                        project=self.project,
                    )

        # Clean the last request records from the KV table
        self._delete_last_request_records(endpoint_ids=endpoint_ids)

        logger.debug(
            "Deleted all model endpoint resources using the V3IO connector",
            project=self.project,
            number_of_endpoints_to_delete=len(endpoint_ids),
        )

    def _delete_last_request_records(self, endpoint_ids: list[str]):
        for endpoint_id in endpoint_ids:
            try:
                self.v3io_client.kv.delete(
                    container=self.container,
                    table=self.last_request_table,
                    key=endpoint_id,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to delete last request record for endpoint '{endpoint_id}'",
                    error=mlrun.errors.err_to_str(e),
                    project=self.project,
                )

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
        get_raw: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
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
        :param get_raw:               Whether to return the request as raw frames rather than a pandas dataframe.
                                      Defaults to False. This can greatly improve performance when a dataframe isn't
                                      needed.

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
            res = self.frames_client.read(
                backend=_TSDB_BE,
                table=table_path,
                start=start,
                end=end,
                columns=columns,
                filter=filter_query,
                aggregation_window=interval,
                aggregators=aggregators,
                step=sliding_window_step,
                get_raw=get_raw,
                **kwargs,
            )
            if get_raw:
                res = list(res)
        except v3io_frames.Error as err:
            if _is_no_schema_error(err):
                return [] if get_raw else pd.DataFrame()
            else:
                raise err

        return res

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
    def _get_v3io_frames_client(
        v3io_container: str, v3io_access_key: str = ""
    ) -> v3io_frames.client.ClientBase:
        return mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=v3io_container,
            token=v3io_access_key,
        )

    @staticmethod
    def _get_endpoint_filter(endpoint_id: Union[str, list[str]]) -> Optional[str]:
        if isinstance(endpoint_id, str):
            return f"endpoint_id=='{endpoint_id}'"
        elif isinstance(endpoint_id, list):
            if len(endpoint_id) > V3IO_FRAMESD_MEPS_LIMIT:
                logger.info(
                    "The number of endpoint ids exceeds the v3io-engine filter-expression limit, "
                    "retrieving all the model endpoints from the db.",
                    limit=V3IO_FRAMESD_MEPS_LIMIT,
                    amount=len(endpoint_id),
                )
                return None
            return f"endpoint_id IN({str(endpoint_id)[1:-1]}) "
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid 'endpoint_id' filter: must be a string or a list, endpoint_id: {endpoint_id}"
            )

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: Literal["metrics", "results"] = "results",
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
        """
        Read metrics OR results from the TSDB and return as a list.
        Note: the type must match the actual metrics in the `metrics` parameter.
        If the type is "results", pass only results in the `metrics` parameter.
        """

        if type == "metrics":
            if with_result_extra_data:
                logger.warning(
                    "The 'with_result_extra_data' parameter is not supported for metrics, just for results",
                    project=self.project,
                    endpoint_id=endpoint_id,
                )
            table_path = self.tables[mm_schemas.V3IOTSDBTables.METRICS]
            name = mm_schemas.MetricData.METRIC_NAME
            columns = [mm_schemas.MetricData.METRIC_VALUE]
            df_handler = self.df_to_metrics_values
        elif type == "results":
            table_path = self.tables[mm_schemas.V3IOTSDBTables.APP_RESULTS]
            name = mm_schemas.ResultData.RESULT_NAME
            columns = [
                mm_schemas.ResultData.RESULT_VALUE,
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.ResultData.RESULT_KIND,
            ]
            if with_result_extra_data:
                columns.append(mm_schemas.ResultData.RESULT_EXTRA_DATA)
            df_handler = self.df_to_results_values
        else:
            raise ValueError(f"Invalid {type = }")

        query = self._get_sql_query(
            endpoint_id=endpoint_id,
            metric_and_app_names=[(metric.app, metric.name) for metric in metrics],
            table_path=table_path,
            name=name,
            columns=columns,
        )

        logger.debug("Querying V3IO TSDB", query=query)

        df: pd.DataFrame = self.frames_client.read(
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
        if not with_result_extra_data and type == "results":
            # Set the extra data to an empty string if it's not requested
            df[mm_schemas.ResultData.RESULT_EXTRA_DATA] = ""

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
        limit: Optional[
            int
        ] = None,  # no effect, just for compatibility with the abstract method
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
            table=mm_schemas.V3IOTSDBTables.PREDICTIONS,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT],
            filter_query=f"endpoint_id=='{endpoint_id}'",
            agg_funcs=agg_funcs,
            sliding_window_step=aggregation_window,
        )

        full_name = get_invocations_fqn(self.project)

        if df.empty:
            return mm_schemas.ModelEndpointMonitoringMetricNoData(
                full_name=full_name,
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )

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
    ) -> dict[str, float]:
        # Get the last request timestamp for each endpoint from the KV table.
        # The result of the query is a list of dictionaries,
        # each dictionary contains the endpoint id and the last request timestamp.
        last_request_timestamps = {}
        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        try:
            if len(endpoint_ids) > V3IO_CLIENT_MEPS_LIMIT:
                logger.warning(
                    "The number of endpoint ids exceeds the v3io-engine filter-expression limit, "
                    "retrieving last request for all the model endpoints from the KV table.",
                    limit=V3IO_CLIENT_MEPS_LIMIT,
                    amount=len(endpoint_ids),
                )

                res = self.v3io_client.kv.new_cursor(
                    container=self.container,
                    table_path=self.last_request_table,
                ).all()
                last_request_timestamps.update(
                    {d["__name"]: d["last_request_timestamp"] for d in res}
                )
            else:
                filter_expression = " OR ".join(
                    [f"__name=='{endpoint_id}'" for endpoint_id in endpoint_ids]
                )
                res = self.v3io_client.kv.new_cursor(
                    container=self.container,
                    table_path=self.last_request_table,
                    filter_expression=filter_expression,
                ).all()
                last_request_timestamps.update(
                    {d["__name"]: d["last_request_timestamp"] for d in res}
                )
        except Exception as e:
            logger.warning(
                "Failed to get last request timestamp from V3IO KV table.",
                err=mlrun.errors.err_to_str(e),
                project=self.project,
                table=self.last_request_table,
            )

        return last_request_timestamps

    def get_drift_status(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_ids)
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._get_start_end(start, end)
        res = self._get_records(
            table=mm_schemas.V3IOTSDBTables.APP_RESULTS,
            start=start,
            end=end,
            columns=[mm_schemas.ResultData.RESULT_STATUS],
            filter_query=filter_query,
            agg_funcs=["max"],
            group_by="endpoint_id",
            get_raw=get_raw,
        )
        if get_raw:
            return res

        df = res
        if not df.empty:
            df.columns = [
                col[len("max(") : -1] if "max(" in col else col for col in df.columns
            ]
        return df.reset_index(drop=True)

    def get_metrics_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        start, end = self._get_start_end(start, end)
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_id)
        df = self._get_records(
            table=mm_schemas.V3IOTSDBTables.METRICS,
            start=start,
            end=end,
            columns=[mm_schemas.MetricData.METRIC_VALUE],
            filter_query=filter_query,
            agg_funcs=["last"],
        )
        if not df.empty:
            df.drop(
                columns=[f"last({mm_schemas.MetricData.METRIC_VALUE})"], inplace=True
            )
        return df.reset_index(drop=True)

    def get_results_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        start, end = self._get_start_end(start, end)
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_id)
        df = self._get_records(
            table=mm_schemas.V3IOTSDBTables.APP_RESULTS,
            start=start,
            end=end,
            columns=[
                mm_schemas.ResultData.RESULT_KIND,
            ],
            filter_query=filter_query,
            agg_funcs=["last"],
        )
        if not df.empty:
            df.rename(
                columns={
                    f"last({mm_schemas.ResultData.RESULT_KIND})": mm_schemas.ResultData.RESULT_KIND
                },
                inplace=True,
            )
        return df.reset_index(drop=True)

    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_ids)
        if filter_query:
            filter_query += f"AND {mm_schemas.EventFieldType.ERROR_TYPE} == '{mm_schemas.EventFieldType.INFER_ERROR}'"
        else:
            filter_query = f"{mm_schemas.EventFieldType.ERROR_TYPE} == '{mm_schemas.EventFieldType.INFER_ERROR}' z"
        start, end = self._get_start_end(start, end)
        res = self._get_records(
            table=mm_schemas.FileTargetKind.ERRORS,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.ERROR_COUNT],
            filter_query=filter_query,
            agg_funcs=["count"],
            get_raw=get_raw,
        )

        if get_raw:
            return res

        df = res
        if not df.empty:
            df.rename(
                columns={
                    f"count({mm_schemas.EventFieldType.ERROR_COUNT})": mm_schemas.EventFieldType.ERROR_COUNT
                },
                inplace=True,
            )
            df.dropna(inplace=True)
        return df.reset_index(drop=True)

    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        filter_query = self._get_endpoint_filter(endpoint_id=endpoint_ids)
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._get_start_end(start, end)
        res = self._get_records(
            table=mm_schemas.V3IOTSDBTables.PREDICTIONS,
            start=start,
            end=end,
            columns=[mm_schemas.EventFieldType.LATENCY],
            filter_query=filter_query,
            agg_funcs=["avg"],
            get_raw=get_raw,
        )

        if get_raw:
            return res

        df = res
        if not df.empty:
            df.dropna(inplace=True)
            df.rename(
                columns={
                    f"avg({mm_schemas.EventFieldType.LATENCY})": f"avg_{mm_schemas.EventFieldType.LATENCY}"
                },
                inplace=True,
            )
        return df.reset_index(drop=True)

    async def add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        project: str,
        run_in_threadpool: Callable,
        metric_list: Optional[list[str]] = None,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Fetch basic metrics from V3IO TSDB and add them to MEP objects.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will
                                       be filled with the relevant basic metrics.
        :param project:                The name of the project.
        :param run_in_threadpool:      A function that runs another function in a thread pool.
        :param metric_list:            List of metrics to include from the time series DB. Defaults to all metrics.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """

        uids = []
        model_endpoint_objects_by_uid = {}
        for model_endpoint_object in model_endpoint_objects:
            uid = model_endpoint_object.metadata.uid
            uids.append(uid)
            model_endpoint_objects_by_uid[uid] = model_endpoint_object

        metric_name_to_function_and_column_name = {
            "error_count": (self.get_error_count, "count(error_count)"),
            "avg_latency": (self.get_avg_latency, "avg(latency)"),
            "result_status": (self.get_drift_status, "max(result_status)"),
        }
        if metric_list is not None:
            for metric_name in list(metric_name_to_function_and_column_name):
                if metric_name not in metric_list:
                    del metric_name_to_function_and_column_name[metric_name]

        metric_name_to_result = {}

        for metric_name, (
            function,
            _,
        ) in metric_name_to_function_and_column_name.items():
            metric_name_to_result[metric_name] = await run_in_threadpool(
                function,
                endpoint_ids=uids,
                get_raw=True,
            )

        def add_metric(
            metric: str,
            column_name: str,
            frames: list,
        ):
            for frame in frames:
                endpoint_ids = frame.column_data("endpoint_id")
                metric_data = frame.column_data(column_name)
                for index, endpoint_id in enumerate(endpoint_ids):
                    mep = model_endpoint_objects_by_uid.get(endpoint_id)
                    value = metric_data[index]
                    if mep and value is not None and not math.isnan(value):
                        setattr(mep.status, metric, value)

        for metric_name, result in metric_name_to_result.items():
            add_metric(
                metric_name,
                metric_name_to_function_and_column_name[metric_name][1],
                result,
            )
        if metric_list is None or "last_request" in metric_list:
            self._enrich_mep_with_last_request(
                model_endpoint_objects_by_uid=model_endpoint_objects_by_uid
            )

        return list(model_endpoint_objects_by_uid.values())

    def _enrich_mep_with_last_request(
        self,
        model_endpoint_objects_by_uid: dict[str, mlrun.common.schemas.ModelEndpoint],
    ):
        last_request_dictionary = self.get_last_request(
            endpoint_ids=list(model_endpoint_objects_by_uid.keys())
        )
        for uid, mep in model_endpoint_objects_by_uid.items():
            # Set the last request timestamp to the MEP object. If not found, keep the existing value from the
            # DB (relevant for batch EP).
            mep.status.last_request = last_request_dictionary.get(
                uid, mep.status.last_request
            )
