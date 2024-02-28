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
import json
from typing import Any

import pandas as pd
import v3io_frames.client
import v3io_frames.errors
from v3io.dataplane import Client as V3IOClient
from v3io_frames.frames_pb2 import IGNORE

import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring import (
    AppResultEvent,
    EventFieldType,
    EventKeyMetrics,
    WriterEvent,
)
from mlrun.model_monitoring.stores.tsdb import TSDBstore
from mlrun.utils import logger

# noinspection PyUnresolvedReferences
from .stream_graph_steps import FilterAndUnpackKeys, ProcessBeforeTSDB

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"


class V3IOTSDBstore(TSDBstore):

    """
    Handles the TSDB operations when the TSDB target is from type V3IO. To manage these operations we use V3IO Frames
    Client that provides API for executing commands on the V3IO TSDB table.
    """

    def __init__(
        self,
        project: str,
        access_key: str = None,
        table: str = None,
        container: str = None,
        v3io_framesd: str = None,
        create_table: bool = False,
    ):
        super().__init__(project=project)
        self.access_key = access_key or mlrun.mlconf.get_v3io_access_key()

        self.table = table
        self.container = container

        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self._frames_client: v3io_frames.client.ClientBase = (
            self._get_v3io_frames_client(self.container)
        )
        self._v3io_client: V3IOClient = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api,
        )

        if create_table:
            self._create_tsdb_table()

    @staticmethod
    def _get_v3io_frames_client(v3io_container: str) -> v3io_frames.client.ClientBase:
        return mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=v3io_container,
        )

    def apply_monitoring_stream_steps(
        self,
        graph,
        tsdb_batching_max_events: int = 10,
        tsdb_batching_timeout_secs: int = 300,
    ):
        """
        Apply TSDB steps on the provided monitoring graph. Throughout these steps, the graph stores live data of
        different key metric dictionaries in TSDB target. This data is being used by the monitoring dashboards in
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
                "ProcessBeforeTSDB", name="ProcessBeforeTSDB", after="sample"
            )

        apply_process_before_tsdb()

        # Steps 13-19: - Unpacked keys from each dictionary and write to TSDB target
        def apply_filter_and_unpacked_keys(name, keys):
            graph.add_step(
                "FilterAndUnpackKeys",
                name=name,
                after="FilterAndUnpackKeys",
                keys=[keys],
            )

        def apply_tsdb_target(name, after):
            graph.add_step(
                "storey.TSDBTarget",
                name=name,
                after=after,
                path=self.table,
                rate="10/m",
                time_col=EventFieldType.TIMESTAMP,
                container=self.container,
                access_key=self.access_key,
                v3io_frames=self.v3io_framesd,
                infer_columns_from_data=True,
                index_cols=[
                    EventFieldType.ENDPOINT_ID,
                    EventFieldType.RECORD_TYPE,
                    EventFieldType.ENDPOINT_TYPE,
                ],
                max_events=tsdb_batching_max_events,
                flush_after_seconds=tsdb_batching_timeout_secs,
                key=EventFieldType.ENDPOINT_ID,
            )

        # Steps 13-14 - unpacked base_metrics dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys1",
            keys=EventKeyMetrics.BASE_METRICS,
        )
        apply_tsdb_target(name="tsdb1", after="FilterAndUnpackKeys1")

        # Steps 15-16 - unpacked endpoint_features dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys2",
            keys=EventKeyMetrics.ENDPOINT_FEATURES,
        )
        apply_tsdb_target(name="tsdb2", after="FilterAndUnpackKeys2")

        # Steps 17-19 - unpacked custom_metrics dictionary. In addition, use storey.Filter remove none values
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys3",
            keys=EventKeyMetrics.CUSTOM_METRICS,
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

    def write_application_event(self, event: AppResultEvent):
        """
        Write a single application result event to the TSDB target.
        """
        event = AppResultEvent(event.copy())
        event[WriterEvent.END_INFER_TIME] = datetime.datetime.fromisoformat(
            event[WriterEvent.END_INFER_TIME]
        )
        del event[WriterEvent.RESULT_EXTRA_DATA]
        try:
            self._frames_client.write(
                backend=_TSDB_BE,
                table=self.table,
                dfs=pd.DataFrame.from_records([event]),
                index_cols=[
                    WriterEvent.END_INFER_TIME,
                    WriterEvent.ENDPOINT_ID,
                    WriterEvent.APPLICATION_NAME,
                    WriterEvent.RESULT_NAME,
                ],
            )
            logger.info("Updated V3IO TSDB successfully", table=self.table)
        except v3io_frames.errors.Error as err:
            logger.warn(
                "Could not write drift measures to TSDB",
                err=err,
                table=self.table,
                event=event,
            )

    def _create_tsdb_table(self) -> None:
        logger.info("Creating table in V3IO TSDB", table=self.table)

        self._frames_client.create(
            backend=_TSDB_BE,
            table=self.table,
            if_exists=IGNORE,
            rate=_TSDB_RATE,
        )

    def update_default_data_drift(
        self,
        endpoint_id: str,
        drift_status: mlrun.common.schemas.model_monitoring.DriftStatus,
        drift_measure: float,
        drift_result: dict[str, dict[str, Any]],
        timestamp: pd.Timestamp,
        stream_container: str,
        stream_path: str,
    ):
        """Update drift results in input stream and TSDB table. The drift results within the input stream are stored
         only if the result indicates on possible drift (or detected drift). Usually the input stream is stored under
        `v3io:///users/pipelines/<project-name>/model-endpoints/log_stream/` while the TSDB table stored under
        `v3io:///users/pipelines/<project-name>/model-endpoints/events/`.
        :param endpoint_id:      The unique id of the model endpoint.
        :param drift_status:     Drift status result. Possible values can be found under DriftStatus enum class.
        :param drift_measure:    The drift result (float) based on the mean of the Total Variance Distance and the
                                 Hellinger distance.
        :param drift_result:     A dictionary that includes the drift results for each feature.
        :param timestamp:        Pandas Timestamp value.
        :param stream_container: Container directory, usually `users`
        :param stream_path:      Input stream full path within the container directory. For storing drift measures,
                                 the path is 'pipelines/<project-name>/model-endpoints/log_stream/'
        """

        if (
            drift_status
            == mlrun.common.schemas.model_monitoring.DriftStatus.POSSIBLE_DRIFT
            or drift_status
            == mlrun.common.schemas.model_monitoring.DriftStatus.DRIFT_DETECTED
        ):
            self._v3io_client.stream.put_records(
                container=stream_container,
                stream_path=stream_path,
                records=[
                    {
                        "data": json.dumps(
                            {
                                "endpoint_id": endpoint_id,
                                "drift_status": drift_status.value,
                                "drift_measure": drift_measure,
                                "drift_per_feature": {**drift_result},
                            }
                        )
                    }
                ],
            )

        # Update the results in tsdb:
        tsdb_drift_measures = {
            "endpoint_id": endpoint_id,
            "timestamp": timestamp,
            "record_type": "drift_measures",
            "tvd_mean": drift_result["tvd_mean"],
            "kld_mean": drift_result["kld_mean"],
            "hellinger_mean": drift_result["hellinger_mean"],
        }

        try:
            self._frames_client.write(
                backend="tsdb",
                table=self.table,
                dfs=pd.DataFrame.from_records([tsdb_drift_measures]),
                index_cols=["timestamp", "endpoint_id", "record_type"],
            )
        except v3io_frames.errors.Error as err:
            logger.warn(
                "Could not write drift measures to TSDB",
                err=err,
                tsdb_path=self.table,
                endpoint=endpoint_id,
            )

    def delete_tsdb_resources(self, table: str = None):
        table = table or self.table
        self._frames_client.delete(
            backend=mlrun.common.schemas.model_monitoring.TimeSeriesTarget.TSDB,
            table=table,
        )

    def get_records(
        self,
        table: str = None,
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
        table = table or self.table
        return self._frames_client.read(
            backend=mlrun.common.schemas.model_monitoring.TimeSeriesTarget.TSDB,
            table=table,
            columns=columns,
            filter=filter_query,
            start=start,
            end=end,
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
                table=self.table,
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
