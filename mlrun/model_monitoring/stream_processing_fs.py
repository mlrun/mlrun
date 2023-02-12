# Copyright 2018 Iguazio
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
import collections
import datetime
import json
import os
import typing

import pandas as pd

# Constants
import storey
import v3io
import v3io.dataplane

import mlrun.config
import mlrun.datastore.targets
import mlrun.feature_store.steps
import mlrun.utils
import mlrun.utils.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.constants import (
    EventFieldType,
    EventKeyMetrics,
    EventLiveStats,
)
from mlrun.utils import logger


# Stream processing code
class EventStreamProcessor:
    def __init__(
        self,
        project: str,
        parquet_batching_max_events: int,
        sample_window: int = 10,
        tsdb_batching_max_events: int = 10,
        tsdb_batching_timeout_secs: int = 60 * 5,  # Default 5 minutes
        parquet_batching_timeout_secs: int = 30 * 60,  # Default 30 minutes
        aggregate_count_windows: typing.Optional[typing.List[str]] = None,
        aggregate_count_period: str = "30s",
        aggregate_avg_windows: typing.Optional[typing.List[str]] = None,
        aggregate_avg_period: str = "30s",
        v3io_access_key: typing.Optional[str] = None,
        v3io_framesd: typing.Optional[str] = None,
        v3io_api: typing.Optional[str] = None,
        model_monitoring_access_key: str = None,
    ):
        self.project = project
        self.sample_window = sample_window
        self.tsdb_batching_max_events = tsdb_batching_max_events
        self.tsdb_batching_timeout_secs = tsdb_batching_timeout_secs
        self.parquet_batching_max_events = parquet_batching_max_events
        self.parquet_batching_timeout_secs = parquet_batching_timeout_secs
        self.aggregate_count_windows = aggregate_count_windows or ["5m", "1h"]
        self.aggregate_count_period = aggregate_count_period
        self.aggregate_avg_windows = aggregate_avg_windows or ["5m", "1h"]
        self.aggregate_avg_period = aggregate_avg_period

        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self.v3io_api = v3io_api or mlrun.mlconf.v3io_api

        self.v3io_access_key = v3io_access_key or os.environ.get("V3IO_ACCESS_KEY")
        self.model_monitoring_access_key = (
            model_monitoring_access_key
            or os.environ.get("MODEL_MONITORING_ACCESS_KEY")
            or self.v3io_access_key
        )
        self.storage_options = dict(
            v3io_access_key=self.model_monitoring_access_key, v3io_api=self.v3io_api
        )

        template = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default

        kv_path = template.format(project=project, kind="endpoints")
        (
            _,
            self.kv_container,
            self.kv_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(kv_path)

        tsdb_path = template.format(project=project, kind="events")
        (
            _,
            self.tsdb_container,
            self.tsdb_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(tsdb_path)
        self.tsdb_path = f"{self.tsdb_container}/{self.tsdb_path}"

        self.parquet_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.user_space.format(
                project=project, kind="parquet"
            )
        )

        logger.info(
            "Initializing model monitoring event stream processor",
            parquet_batching_max_events=self.parquet_batching_max_events,
            v3io_access_key=self.v3io_access_key,
            model_monitoring_access_key=self.model_monitoring_access_key,
            default_store_prefix=mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default,
            user_space_store_prefix=mlrun.mlconf.model_endpoint_monitoring.store_prefixes.user_space,
            v3io_api=self.v3io_api,
            v3io_framesd=self.v3io_framesd,
            kv_container=self.kv_container,
            kv_path=self.kv_path,
            tsdb_container=self.tsdb_container,
            tsdb_path=self.tsdb_path,
            parquet_path=self.parquet_path,
        )

    def apply_monitoring_serving_graph(self, fn):
        """
        Apply monitoring serving graph to a given serving function. The following serving graph includes about 20 steps
        of different operations that are executed on the events from the model server. Each event has
        metadata (function_uri, timestamp, class, etc.) but also inputs and predictions from the model server.
        Throughout the serving graph, the results are written to 3 different databases:
        1. KV (steps 7-9): Stores metadata and stats about the average latency and the amount of predictions over time
           per endpoint. for example the amount of predictions of endpoint x in the last 5 min. This data is used by
           the monitoring dashboards in grafana. Please note that the KV table, which can be found under
           v3io:///users/pipelines/project-name/model-endpoints/endpoints/ also contains data on the model endpoint
            from other processes, such as current_stats that is being calculated by the monitoring batch job
            process.
        2. TSDB (steps 12-18): Stores live data of different key metric dictionaries in tsdb target. Results can be
           found under v3io:///users/pipelines/project-name/model-endpoints/events/. At the moment, this part supports
           3 different key metric dictionaries: base_metrics (average latency and predictions over time),
           endpoint_features (Prediction and feature names and values), and custom_metrics (user-defined metrics).
           This data is also being used by the monitoring dashboards in grafana.
        3. Parquet (steps 19-20): This Parquet file includes the required data for the model monitoring batch job
           that run every hour by default. The parquet target can be found under
           v3io:///projects/{project}/model-endpoints/.

        :param fn: A serving function.
        """

        graph = fn.set_topology("flow")

        # Step 1 - Process endpoint event: splitting into sub-events and validate event data
        def apply_process_endpoint_event():
            graph.add_step(
                "ProcessEndpointEvent",
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                v3io_access_key=self.v3io_access_key,
                full_event=True,
                project=self.project,
            )

        apply_process_endpoint_event()

        # Steps 2,3 - Applying Storey operations of filtering and flatten
        def apply_storey_filter_and_flatmap():
            # Remove none values from each event
            graph.add_step(
                "storey.Filter",
                "filter_none",
                _fn="(event is not None)",
                after="ProcessEndpointEvent",
            )

            # flatten the events
            graph.add_step(
                "storey.FlatMap", "flatten_events", _fn="(event)", after="filter_none"
            )

        apply_storey_filter_and_flatmap()

        # Step 4 - Validating feature names and map each feature to its value
        def apply_map_feature_names():
            graph.add_step(
                "MapFeatureNames",
                name="MapFeatureNames",
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                access_key=self.v3io_access_key,
                infer_columns_from_data=True,
                after="flatten_events",
            )

        apply_map_feature_names()

        # Step 5 - Calculate number of predictions and average latency
        def apply_storey_aggregations():
            # Step 5.1 - Calculate number of predictions for each window (5 min and 1 hour by default)
            graph.add_step(
                class_name="storey.AggregateByKey",
                aggregates=[
                    {
                        "name": EventFieldType.PREDICTIONS,
                        "column": EventFieldType.ENDPOINT_ID,
                        "operations": ["count"],
                        "windows": self.aggregate_count_windows,
                        "period": self.aggregate_count_period,
                    }
                ],
                name=EventFieldType.PREDICTIONS,
                after="MapFeatureNames",
                step_name="Aggregates",
                table=".",
                v3io_access_key=self.v3io_access_key,
            )
            # Step 5.2 - Calculate average latency time for each window (5 min and 1 hour by default)
            graph.add_step(
                class_name="storey.AggregateByKey",
                aggregates=[
                    {
                        "name": EventFieldType.LATENCY,
                        "column": EventFieldType.LATENCY,
                        "operations": ["avg"],
                        "windows": self.aggregate_avg_windows,
                        "period": self.aggregate_avg_period,
                    }
                ],
                name=EventFieldType.LATENCY,
                after=EventFieldType.PREDICTIONS,
                table=".",
                v3io_access_key=self.v3io_access_key,
            )

        apply_storey_aggregations()

        # Step 6 - Emits the event in window size of events based on sample_window size (10 by default)
        def apply_storey_sample_window():
            graph.add_step(
                "storey.steps.SampleWindow",
                name="sample",
                after=EventFieldType.LATENCY,
                window_size=self.sample_window,
                key=EventFieldType.ENDPOINT_ID,
                v3io_access_key=self.v3io_access_key,
            )

        apply_storey_sample_window()

        # Steps 7-9 - KV branch
        # Step 7 - Filter relevant keys from the event before writing the data into KV
        def apply_process_before_kv():
            graph.add_step("ProcessBeforeKV", name="ProcessBeforeKV", after="sample")

        apply_process_before_kv()

        # Step 8 - Write the filtered event to KV table. At this point, the serving graph updates the stats
        # about average latency and the amount of predictions over time
        def apply_write_to_kv():
            graph.add_step(
                "WriteToKV",
                name="WriteToKV",
                after="ProcessBeforeKV",
                container=self.kv_container,
                table=self.kv_path,
                v3io_access_key=self.v3io_access_key,
            )

        apply_write_to_kv()

        # Step 9 - Apply infer_schema on the KB table for generating schema file
        # which will be used by Grafana monitoring dashboards
        def apply_infer_schema():
            graph.add_step(
                "InferSchema",
                name="InferSchema",
                after="WriteToKV",
                v3io_access_key=self.v3io_access_key,
                v3io_framesd=self.v3io_framesd,
                container=self.kv_container,
                table=self.kv_path,
            )

        apply_infer_schema()

        # Steps 11-18 - TSDB branch
        # Step 11 - Before writing data to TSDB, create dictionary of 2-3 dictionaries that contains
        # stats and details about the events
        def apply_process_before_tsdb():
            graph.add_step(
                "ProcessBeforeTSDB", name="ProcessBeforeTSDB", after="sample"
            )

        apply_process_before_tsdb()

        # Steps 12-18: - Unpacked keys from each dictionary and write to TSDB target
        def apply_filter_and_unpacked_keys(name, keys):
            graph.add_step(
                "FilterAndUnpackKeys",
                name=name,
                after="ProcessBeforeTSDB",
                keys=[keys],
            )

        def apply_tsdb_target(name, after):
            graph.add_step(
                "storey.TSDBTarget",
                name=name,
                after=after,
                path=self.tsdb_path,
                rate="10/m",
                time_col=EventFieldType.TIMESTAMP,
                container=self.tsdb_container,
                access_key=self.v3io_access_key,
                v3io_frames=self.v3io_framesd,
                infer_columns_from_data=True,
                index_cols=[
                    EventFieldType.ENDPOINT_ID,
                    EventFieldType.RECORD_TYPE,
                ],
                max_events=self.tsdb_batching_max_events,
                flush_after_seconds=self.tsdb_batching_timeout_secs,
                key=EventFieldType.ENDPOINT_ID,
            )

        # Steps 12-13 - unpacked base_metrics dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys1",
            keys=EventKeyMetrics.BASE_METRICS,
        )
        apply_tsdb_target(name="tsdb1", after="FilterAndUnpackKeys1")

        # Steps 14-15 - unpacked endpoint_features dictionary
        apply_filter_and_unpacked_keys(
            name="FilterAndUnpackKeys2",
            keys=EventKeyMetrics.ENDPOINT_FEATURES,
        )
        apply_tsdb_target(name="tsdb2", after="FilterAndUnpackKeys2")

        # Steps 16-18 - unpacked custom_metrics dictionary. In addition, use storey.Filter remove none values
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

        # Steps 19-20 - Parquet branch
        # Step 19 - Filter and validate different keys before writing the data to Parquet target
        def apply_process_before_parquet():
            graph.add_step(
                "ProcessBeforeParquet",
                name="ProcessBeforeParquet",
                after="MapFeatureNames",
                _fn="(event)",
            )

        apply_process_before_parquet()

        # Step 20 - Write the Parquet target file, partitioned by key (endpoint_id) and time.
        def apply_parquet_target():
            graph.add_step(
                "storey.ParquetTarget",
                name="ParquetTarget",
                after="ProcessBeforeParquet",
                graph_shape="cylinder",
                path=self.parquet_path,
                storage_options=self.storage_options,
                max_events=self.parquet_batching_max_events,
                flush_after_seconds=self.parquet_batching_timeout_secs,
                attributes={"infer_columns_from_data": True},
                index_cols=[EventFieldType.ENDPOINT_ID],
                key_bucketing_number=0,
                time_partitioning_granularity="hour",
                partition_cols=["$key", "$year", "$month", "$day", "$hour"],
            )

        apply_parquet_target()


class ProcessBeforeKV(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Filter relevant keys from the event before writing the data to KV table (in WriteToKV step). Note that in KV
        we only keep metadata (function_uri, model_class, etc.) and stats about the average latency and the number
        of predictions (per 5min and 1hour).

        :returns: A filtered event as a dictionary which will be written to KV table in the next step.
        """
        super().__init__(**kwargs)

    def do(self, event):
        # Compute prediction per second
        event[EventLiveStats.PREDICTIONS_PER_SECOND] = (
            float(event[EventLiveStats.PREDICTIONS_COUNT_5M]) / 300
        )
        # Filter relevant keys
        e = {
            k: event[k]
            for k in [
                EventFieldType.FUNCTION_URI,
                EventFieldType.MODEL,
                EventFieldType.MODEL_CLASS,
                EventFieldType.TIMESTAMP,
                EventFieldType.ENDPOINT_ID,
                EventFieldType.LABELS,
                EventFieldType.UNPACKED_LABELS,
                EventLiveStats.LATENCY_AVG_5M,
                EventLiveStats.LATENCY_AVG_1H,
                EventLiveStats.PREDICTIONS_PER_SECOND,
                EventLiveStats.PREDICTIONS_COUNT_5M,
                EventLiveStats.PREDICTIONS_COUNT_1H,
                EventFieldType.FIRST_REQUEST,
                EventFieldType.LAST_REQUEST,
                EventFieldType.ERROR_COUNT,
            ]
        }
        # Unpack labels dictionary
        e = {
            **e.pop(EventFieldType.UNPACKED_LABELS, {}),
            **e,
        }
        # Write labels to kv as json string to be presentable later
        e[EventFieldType.LABELS] = json.dumps(e[EventFieldType.LABELS])
        return e


class ProcessBeforeTSDB(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Process the data before writing to TSDB. This step creates a dictionary that includes 3 different dictionaries
        that each one of them contains important details and stats about the events:
        1. base_metrics: stats about the average latency and the amount of predictions over time. It is based on
           storey.AggregateByKey which was executed in step 5.
        2. endpoint_features: feature names and values along with the prediction names and value.
        3. custom_metric (opt): optional metrics provided by the user.

        :returns: Dictionary of 2-3 dictionaries that contains stats and details about the events.

        """
        super().__init__(**kwargs)

    def do(self, event):
        # Compute prediction per second
        event[EventLiveStats.PREDICTIONS_PER_SECOND] = (
            float(event[EventLiveStats.PREDICTIONS_COUNT_5M]) / 300
        )
        base_fields = [
            EventFieldType.TIMESTAMP,
            EventFieldType.ENDPOINT_ID,
        ]

        # Getting event timestamp and endpoint_id
        base_event = {k: event[k] for k in base_fields}
        base_event[EventFieldType.TIMESTAMP] = pd.to_datetime(
            base_event[EventFieldType.TIMESTAMP],
            format=EventFieldType.TIME_FORMAT,
        )

        # base_metrics includes the stats about the average latency and the amount of predictions over time
        base_metrics = {
            EventFieldType.RECORD_TYPE: EventKeyMetrics.BASE_METRICS,
            EventLiveStats.PREDICTIONS_PER_SECOND: event[
                EventLiveStats.PREDICTIONS_PER_SECOND
            ],
            EventLiveStats.PREDICTIONS_COUNT_5M: event[
                EventLiveStats.PREDICTIONS_COUNT_5M
            ],
            EventLiveStats.PREDICTIONS_COUNT_1H: event[
                EventLiveStats.PREDICTIONS_COUNT_1H
            ],
            EventLiveStats.LATENCY_AVG_5M: event[EventLiveStats.LATENCY_AVG_5M],
            EventLiveStats.LATENCY_AVG_1H: event[EventLiveStats.LATENCY_AVG_1H],
            **base_event,
        }

        # endpoint_features includes the event values of each feature and prediction
        endpoint_features = {
            EventFieldType.RECORD_TYPE: EventKeyMetrics.ENDPOINT_FEATURES,
            **event[EventFieldType.NAMED_PREDICTIONS],
            **event[EventFieldType.NAMED_FEATURES],
            **base_event,
        }
        # Create a dictionary that includes both base_metrics and endpoint_features
        processed = {
            EventKeyMetrics.BASE_METRICS: base_metrics,
            EventKeyMetrics.ENDPOINT_FEATURES: endpoint_features,
        }

        # If metrics provided, add another dictionary if custom_metrics values
        if event[EventFieldType.METRICS]:
            processed[EventKeyMetrics.CUSTOM_METRICS] = {
                EventFieldType.RECORD_TYPE: EventKeyMetrics.CUSTOM_METRICS,
                **event[EventFieldType.METRICS],
                **base_event,
            }

        return processed


class ProcessBeforeParquet(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Process the data before writing to Parquet file. In this step, unnecessary keys will be removed while possible
        missing keys values will be set to None.

        :returns: Event dictionary with filtered data for the Parquet target.

        """
        super().__init__(**kwargs)

    def do(self, event):
        logger.info("ProcessBeforeParquet1", event=event)
        # Remove the following keys from the event
        for key in [
            EventFieldType.UNPACKED_LABELS,
            EventFieldType.FEATURES,
            EventFieldType.NAMED_FEATURES,
        ]:
            event.pop(key, None)

        # Split entities dictionary to separate dictionaries within the event
        value = event.get("entities")
        if value is not None:
            event = {**value, **event}

        # Validate that the following keys exist
        for key in [
            EventFieldType.LABELS,
            EventFieldType.METRICS,
            EventFieldType.ENTITIES,
        ]:
            if not event.get(key):
                event[key] = None
        logger.info("ProcessBeforeParquet2", event=event)
        return event


class ProcessEndpointEvent(mlrun.feature_store.steps.MapClass):
    def __init__(
        self,
        kv_container: str,
        kv_path: str,
        v3io_access_key: str,
        **kwargs,
    ):
        """
        Process event or batch of events as part of the first step of the monitoring serving graph. It includes
        Adding important details to the event such as endpoint_id, handling errors coming from the stream, Validation
        of event data such as inputs and outputs, and splitting model event into sub-events.

        :param kv_container:    Name of the container that will be used to retrieve the endpoint id. For model
                                endpoints it is usually 'users'.
        :param kv_path:         KV table path that will be used to retrieve the endpoint id. For model endpoints
                                it is usually pipelines/project-name/model-endpoints/endpoints/
        :param v3io_access_key: Access key with permission to read from a KV table.
        :param project:         Project name.


        :returns: A Storey event object which is the basic unit of data in Storey. Note that the next steps of
                  the monitoring serving graph are based on Storey operations.

        """
        super().__init__(**kwargs)
        self.kv_container: str = kv_container
        self.kv_path: str = kv_path
        self.v3io_access_key: str = v3io_access_key

        # First and last requests timestamps (value) of each endpoint (key)
        self.first_request: typing.Dict[str, str] = dict()
        self.last_request: typing.Dict[str, str] = dict()

        # Number of errors (value) per endpoint (key)
        self.error_count: typing.Dict[str, int] = collections.defaultdict(int)

        # Set of endpoints in the current events
        self.endpoints: typing.Set[str] = set()

    def do(self, full_event):
        event = full_event.body

        # Getting model version and function uri from event
        # and use them for retrieving the endpoint_id
        function_uri = event.get(EventFieldType.FUNCTION_URI)
        if not is_not_none(function_uri, [EventFieldType.FUNCTION_URI]):
            return None

        model = event.get(EventFieldType.MODEL)
        if not is_not_none(model, [EventFieldType.MODEL]):
            return None

        version = event.get(EventFieldType.VERSION)
        versioned_model = f"{model}:{version}" if version else f"{model}:latest"

        endpoint_id = mlrun.utils.model_monitoring.create_model_endpoint_id(
            function_uri=function_uri,
            versioned_model=versioned_model,
        )

        endpoint_id = str(endpoint_id)

        event[EventFieldType.VERSIONED_MODEL] = versioned_model
        event[EventFieldType.ENDPOINT_ID] = endpoint_id

        # In case this process fails, resume state from existing record
        self.resume_state(endpoint_id)

        # Handle errors coming from stream
        found_errors = self.handle_errors(endpoint_id, event)
        if found_errors:
            return None

        # Validate event fields
        model_class = event.get("model_class") or event.get("class")
        timestamp = event.get("when")
        request_id = event.get("request", {}).get("id") or event.get("resp", {}).get(
            "id"
        )
        latency = event.get("microsec")
        features = event.get("request", {}).get("inputs")
        predictions = event.get("resp", {}).get("outputs")

        if not self.is_valid(
            endpoint_id,
            is_not_none,
            timestamp,
            ["when"],
        ):
            return None

        if endpoint_id not in self.first_request:
            self.first_request[endpoint_id] = timestamp
        self.last_request[endpoint_id] = timestamp

        if not self.is_valid(
            endpoint_id,
            is_not_none,
            request_id,
            ["request", "id"],
        ):
            return None
        if not self.is_valid(
            endpoint_id,
            is_not_none,
            latency,
            ["microsec"],
        ):
            return None
        if not self.is_valid(
            endpoint_id,
            is_not_none,
            features,
            ["request", "inputs"],
        ):
            return None
        if not self.is_valid(
            endpoint_id,
            is_not_none,
            predictions,
            ["resp", "outputs"],
        ):
            return None

        # Get labels from event (if exist)
        unpacked_labels = {
            f"_{k}": v for k, v in event.get(EventFieldType.LABELS, {}).items()
        }

        # Adjust timestamp format
        timestamp = datetime.datetime.strptime(timestamp[:-6], "%Y-%m-%d %H:%M:%S.%f")

        # Separate each model invocation into sub events that will be stored as dictionary
        # in list of events. This list will be used as the body for the storey event.
        events = []
        for i, (feature, prediction) in enumerate(zip(features, predictions)):
            # Validate that inputs are based on numeric values
            if not self.is_valid(
                endpoint_id,
                self.is_list_of_numerics,
                feature,
                ["request", "inputs", f"[{i}]"],
            ):
                return None

            if not isinstance(prediction, list):
                prediction = [prediction]

            events.append(
                {
                    EventFieldType.FUNCTION_URI: function_uri,
                    EventFieldType.MODEL: versioned_model,
                    EventFieldType.MODEL_CLASS: model_class,
                    EventFieldType.TIMESTAMP: timestamp,
                    EventFieldType.ENDPOINT_ID: endpoint_id,
                    EventFieldType.REQUEST_ID: request_id,
                    EventFieldType.LATENCY: latency,
                    EventFieldType.FEATURES: feature,
                    EventFieldType.PREDICTION: prediction,
                    EventFieldType.FIRST_REQUEST: self.first_request[endpoint_id],
                    EventFieldType.LAST_REQUEST: self.last_request[endpoint_id],
                    EventFieldType.ERROR_COUNT: self.error_count[endpoint_id],
                    EventFieldType.LABELS: event.get(EventFieldType.LABELS, {}),
                    EventFieldType.METRICS: event.get(EventFieldType.METRICS, {}),
                    EventFieldType.ENTITIES: event.get("request", {}).get(
                        EventFieldType.ENTITIES, {}
                    ),
                    EventFieldType.UNPACKED_LABELS: unpacked_labels,
                }
            )

        # Create a storey event object with list of events, based on endpoint_id which will be used
        # in the upcoming steps
        storey_event = storey.Event(body=events, key=endpoint_id)
        return storey_event

    def is_list_of_numerics(
        self,
        field: typing.List[typing.Union[int, float, dict, list]],
        dict_path: typing.List[str],
    ):
        if all(isinstance(x, int) or isinstance(x, float) for x in field):
            return True
        logger.error(
            f"List does not consist of only numeric values: {field} [Event -> {','.join(dict_path)}]"
        )
        return False

    def resume_state(self, endpoint_id):
        # Make sure process is resumable, if process fails for any reason, be able to pick things up close to where we
        # left them
        if endpoint_id not in self.endpoints:
            logger.info("Trying to resume state", endpoint_id=endpoint_id)
            endpoint_record = get_endpoint_record(
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                endpoint_id=endpoint_id,
                access_key=self.v3io_access_key,
            )

            # If model endpoint found, validate first_request and error_count values
            if endpoint_record:
                first_request = endpoint_record.get(EventFieldType.FIRST_REQUEST)
                if first_request:
                    self.first_request[endpoint_id] = first_request
                error_count = endpoint_record.get(EventFieldType.ERROR_COUNT)
                if error_count:
                    self.error_count[endpoint_id] = error_count

            # add endpoint to endpoints set
            self.endpoints.add(endpoint_id)

    def is_valid(
        self,
        endpoint_id: str,
        validation_function,
        field: typing.Any,
        dict_path: typing.List[str],
    ):
        if validation_function(field, dict_path):
            return True
        self.error_count[endpoint_id] += 1
        return False

    def handle_errors(self, endpoint_id, event) -> bool:
        if "error" in event:
            self.error_count[endpoint_id] += 1
            return True

        return False


def is_not_none(field: typing.Any, dict_path: typing.List[str]):
    if field is not None:
        return True
    logger.error(
        f"Expected event field is missing: {field} [Event -> {','.join(dict_path)}]"
    )
    return False


class FilterAndUnpackKeys(mlrun.feature_store.steps.MapClass):
    def __init__(self, keys, **kwargs):
        """
        Create unpacked event dictionary based on provided key metrics (base_metrics, endpoint_features,
        or custom_metric). Please note that the next step of the TSDB target requires an unpacked dictionary.

        :param keys: list of key metrics.

        :returns: An unpacked dictionary of event filtered by the provided key metrics.
        """
        super().__init__(**kwargs)
        self.keys = keys

    def do(self, event):
        # Keep only the relevant dictionary based on the provided keys
        new_event = {}
        for key in self.keys:
            if key in event:
                new_event[key] = event[key]

        # Create unpacked dictionary
        unpacked = {}
        for key in new_event.keys():
            if key in self.keys:
                unpacked = {**unpacked, **new_event[key]}
            else:
                unpacked[key] = new_event[key]
        return unpacked if unpacked else None


class MapFeatureNames(mlrun.feature_store.steps.MapClass):
    def __init__(
        self,
        kv_container: str,
        kv_path: str,
        access_key: str,
        infer_columns_from_data: bool = False,
        **kwargs,
    ):
        """
        Validating feature names and label columns and map each feature to its value. In the end of this step,
        the event should have key-value pairs of (feature name: feature value).

        :param kv_container:            Name of the container that will be used to retrieve the endpoint id. For model
                                        endpoints it is usually 'users'.
        :param kv_path:                 KV table path that will be used to retrieve the endpoint id. For model endpoints
                                        it is usually pipelines/project-name/model-endpoints/endpoints/
        :param v3io_access_key:         Access key with permission to read from a KV table.
        :param infer_columns_from_data: If true and features or labels names were not found, then try to
                                        retrieve them from data that was stored in the previous events of
                                        the current process. This data can be found under self.feature_names and
                                        self.label_columns.


        :returns: A single event as a dictionary that includes metadata (endpoint_id, model_class, etc.) and also
                  feature names and values (as well as the prediction results).
        """
        super().__init__(**kwargs)
        self.kv_container = kv_container
        self.kv_path = kv_path
        self.access_key = access_key
        self._infer_columns_from_data = infer_columns_from_data

        # Dictionaries that will be used in case features names
        # and labels columns were not found in the current event
        self.feature_names = {}
        self.label_columns = {}

    def _infer_feature_names_from_data(self, event):
        for endpoint_id in self.feature_names:
            if len(self.feature_names[endpoint_id]) >= len(
                event[EventFieldType.FEATURES]
            ):
                return self.feature_names[endpoint_id]
        return None

    def _infer_label_columns_from_data(self, event):
        for endpoint_id in self.label_columns:
            if len(self.label_columns[endpoint_id]) >= len(
                event[EventFieldType.PREDICTION]
            ):
                return self.label_columns[endpoint_id]
        return None

    def do(self, event: typing.Dict):
        endpoint_id = event[EventFieldType.ENDPOINT_ID]

        # Get feature names and label columns
        if endpoint_id not in self.feature_names:
            endpoint_record = get_endpoint_record(
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                endpoint_id=endpoint_id,
                access_key=self.access_key,
            )
            feature_names = endpoint_record.get(EventFieldType.FEATURE_NAMES)
            feature_names = json.loads(feature_names) if feature_names else None

            label_columns = endpoint_record.get(EventFieldType.LABEL_NAMES)
            label_columns = json.loads(label_columns) if label_columns else None

            # Ff feature names were not found,
            # try to retrieve them from the previous events of the current process
            if not feature_names and self._infer_columns_from_data:
                feature_names = self._infer_feature_names_from_data(event)

            if not feature_names:
                logger.warn(
                    "Feature names are not initialized, they will be automatically generated",
                    endpoint_id=endpoint_id,
                )
                feature_names = [
                    f"f{i}" for i, _ in enumerate(event[EventFieldType.FEATURES])
                ]

                # Update the endpoint record with the generated features
                mlrun.utils.v3io_clients.get_v3io_client().kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    access_key=self.access_key,
                    key=event[EventFieldType.ENDPOINT_ID],
                    attributes={
                        EventFieldType.FEATURE_NAMES: json.dumps(feature_names)
                    },
                    raise_for_status=v3io.dataplane.RaiseForStatus.always,
                )

            # Similar process with label columns
            if not label_columns and self._infer_columns_from_data:
                label_columns = self._infer_label_columns_from_data(event)

            if not label_columns:
                logger.warn(
                    "label column names are not initialized, they will be automatically generated",
                    endpoint_id=endpoint_id,
                )
                label_columns = [
                    f"p{i}" for i, _ in enumerate(event[EventFieldType.PREDICTION])
                ]
                mlrun.utils.v3io_clients.get_v3io_client().kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    access_key=self.access_key,
                    key=event[EventFieldType.ENDPOINT_ID],
                    attributes={
                        EventFieldType.LABEL_COLUMNS: json.dumps(label_columns)
                    },
                    raise_for_status=v3io.dataplane.RaiseForStatus.always,
                )

            self.label_columns[endpoint_id] = label_columns
            self.feature_names[endpoint_id] = feature_names

            logger.info(
                "Label columns", endpoint_id=endpoint_id, label_columns=label_columns
            )
            logger.info(
                "Feature names", endpoint_id=endpoint_id, feature_names=feature_names
            )

        # Add feature_name:value pairs along with a mapping dictionary of all of these pairs
        feature_names = self.feature_names[endpoint_id]
        feature_values = event[EventFieldType.FEATURES]
        self._map_dictionary_values(
            event=event,
            named_iters=feature_names,
            values_iters=feature_values,
            mapping_dictionary=EventFieldType.NAMED_FEATURES,
        )

        # Add label_name:value pairs along with a mapping dictionary of all of these pairs
        label_names = self.label_columns[endpoint_id]
        label_values = event[EventFieldType.PREDICTION]
        self._map_dictionary_values(
            event=event,
            named_iters=label_names,
            values_iters=label_values,
            mapping_dictionary=EventFieldType.NAMED_PREDICTIONS,
        )

        logger.info("Mapped event", event=event)
        return event

    @staticmethod
    def _map_dictionary_values(
        event: typing.Dict,
        named_iters: typing.List,
        values_iters: typing.List,
        mapping_dictionary: str,
    ):
        """Adding name-value pairs to event dictionary based on two provided lists of names and values. These pairs
        will be used mainly for the Parquet target file. In addition, this function creates a new mapping dictionary of
        these pairs which will be unpacked in ProcessBeforeTSDB step

        :param event:               A dictionary that includes details about the current event such as endpoint_id
                                    and input names and values.
        :param named_iters:         List of names to match to the list of values.
        :param values_iters:        List of values to match to the list of names.
        :param mapping_dictionary:  Name of the new dictionary that will be stored in the current event. The new
                                    dictionary includes name-value pairs based on the provided named_iters and
                                    values_iters lists.

        """
        event[mapping_dictionary] = {}
        for name, value in zip(named_iters, values_iters):
            event[name] = value
            event[mapping_dictionary][name] = value


class WriteToKV(mlrun.feature_store.steps.MapClass):
    def __init__(self, container: str, table: str, v3io_access_key: str, **kwargs):
        """
        Writes the event to KV table. Note that the event at this point includes metadata and stats about the
        average latency and the amount of predictions over time. This data will be used in the monitoring dashboards
        such as "Model Monitoring - Performance" which can be found in Grafana.

        :param kv_container:            Name of the container that will be used to retrieve the endpoint id. For model
                                        endpoints it is usually 'users'.
        :param table:                   KV table path that will be used to retrieve the endpoint id. For model endpoints
                                        it is usually pipelines/project-name/model-endpoints/endpoints/.
        :param v3io_access_key:         Access key with permission to read from a KV table.

        :returns: Event as a dictionary (without any changes) for the next step (InferSchema).
        """
        super().__init__(**kwargs)
        self.container = container
        self.table = table
        self.v3io_access_key = v3io_access_key

    def do(self, event: typing.Dict):
        mlrun.utils.v3io_clients.get_v3io_client().kv.update(
            container=self.container,
            table_path=self.table,
            key=event[EventFieldType.ENDPOINT_ID],
            attributes=event,
            access_key=self.v3io_access_key,
        )
        return event


class InferSchema(mlrun.feature_store.steps.MapClass):
    def __init__(
        self,
        v3io_access_key: str,
        v3io_framesd: str,
        container: str,
        table: str,
        **kwargs,
    ):
        """
        Apply infer_schema on the kv table which generates the schema file.
        Grafana monitoring dashboards use this schema to query the relevant stats.

        :param v3io_access_key:         Access key with permission to a KV table.
        :v3io_framesd:                  path to v3io frames.
        :param container:               Name of the container that will be used to retrieve the endpoint id. For model
                                        endpoints it is usually 'users'.
        :param table:                   KV table path that will be used to retrieve the endpoint id. For model endpoints
                                        it is usually pipelines/project-name/model-endpoints/endpoints/.

        """
        super().__init__(**kwargs)
        self.container = container
        self.v3io_access_key = v3io_access_key
        self.v3io_framesd = v3io_framesd
        self.table = table
        self.keys = set()

    def do(self, event: typing.Dict):
        key_set = set(event.keys())
        if not key_set.issubset(self.keys):
            self.keys.update(key_set)
            # Apply infer_schema on the kv table for generating the schema file
            mlrun.utils.v3io_clients.get_frames_client(
                token=self.v3io_access_key,
                container=self.container,
                address=self.v3io_framesd,
            ).execute(backend="kv", table=self.table, command="infer_schema")
        return event


def get_endpoint_record(
    kv_container: str, kv_path: str, endpoint_id: str, access_key: str
) -> typing.Optional[dict]:
    logger.info(
        "Grabbing endpoint data",
        container=kv_container,
        table_path=kv_path,
        key=endpoint_id,
    )
    try:
        endpoint_record = (
            mlrun.utils.v3io_clients.get_v3io_client()
            .kv.get(
                container=kv_container,
                table_path=kv_path,
                key=endpoint_id,
                access_key=access_key,
                raise_for_status=v3io.dataplane.RaiseForStatus.always,
            )
            .output.item
        )
        return endpoint_record
    except Exception:
        return None
