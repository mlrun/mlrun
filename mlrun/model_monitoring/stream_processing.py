# Copyright 2023 Iguazio
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

import collections
import datetime
import json
import os
import typing

import storey

import mlrun
import mlrun.common.model_monitoring.helpers
import mlrun.config
import mlrun.datastore.targets
import mlrun.feature_store as fstore
import mlrun.feature_store.steps
import mlrun.model_monitoring.db
import mlrun.serving.states
import mlrun.utils
from mlrun.common.schemas.model_monitoring.constants import (
    EndpointType,
    EventFieldType,
    EventKeyMetrics,
    EventLiveStats,
    FileTargetKind,
    ModelEndpointTarget,
    ProjectSecretKeys,
)
from mlrun.model_monitoring.db import StoreBase, TSDBConnector
from mlrun.utils import logger


# Stream processing code
class EventStreamProcessor:
    def __init__(
        self,
        project: str,
        parquet_batching_max_events: int,
        parquet_batching_timeout_secs: int,
        parquet_target: str,
        aggregate_windows: typing.Optional[list[str]] = None,
        aggregate_period: str = "5m",
        model_monitoring_access_key: typing.Optional[str] = None,
    ):
        # General configurations, mainly used for the storey steps in the future serving graph
        self.project = project
        self.aggregate_windows = aggregate_windows or ["5m", "1h"]
        self.aggregate_period = aggregate_period

        # Parquet path and configurations
        self.parquet_path = parquet_target
        self.parquet_batching_max_events = parquet_batching_max_events
        self.parquet_batching_timeout_secs = parquet_batching_timeout_secs

        logger.info(
            "Initializing model monitoring event stream processor",
            parquet_path=self.parquet_path,
            parquet_batching_max_events=self.parquet_batching_max_events,
        )

        self.storage_options = None
        self.tsdb_configurations = {}
        if not mlrun.mlconf.is_ce_mode():
            self._initialize_v3io_configurations(
                model_monitoring_access_key=model_monitoring_access_key
            )
        elif self.parquet_path.startswith("s3://"):
            self.storage_options = mlrun.mlconf.get_s3_storage_options()

    def _initialize_v3io_configurations(
        self,
        tsdb_batching_max_events: int = 10,
        tsdb_batching_timeout_secs: int = 60 * 5,  # Default 5 minutes
        v3io_access_key: typing.Optional[str] = None,
        v3io_framesd: typing.Optional[str] = None,
        v3io_api: typing.Optional[str] = None,
        model_monitoring_access_key: typing.Optional[str] = None,
    ):
        # Get the V3IO configurations
        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self.v3io_api = v3io_api or mlrun.mlconf.v3io_api

        self.v3io_access_key = v3io_access_key or os.environ.get("V3IO_ACCESS_KEY")
        self.model_monitoring_access_key = (
            model_monitoring_access_key
            or os.environ.get(ProjectSecretKeys.ACCESS_KEY)
            or self.v3io_access_key
        )
        self.storage_options = dict(
            v3io_access_key=self.model_monitoring_access_key, v3io_api=self.v3io_api
        )

        # KV path
        kv_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project, kind=FileTargetKind.ENDPOINTS
        )
        (
            _,
            self.kv_container,
            self.kv_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            kv_path
        )

        # TSDB path and configurations
        tsdb_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project, kind=FileTargetKind.EVENTS
        )
        (
            _,
            self.tsdb_container,
            self.tsdb_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            tsdb_path
        )

        self.tsdb_path = f"{self.tsdb_container}/{self.tsdb_path}"
        self.tsdb_batching_max_events = tsdb_batching_max_events
        self.tsdb_batching_timeout_secs = tsdb_batching_timeout_secs

    def apply_monitoring_serving_graph(
        self,
        fn: mlrun.runtimes.ServingRuntime,
        tsdb_connector: TSDBConnector,
        endpoint_store: StoreBase,
    ) -> None:
        """
        Apply monitoring serving graph to a given serving function. The following serving graph includes about 4 main
        parts that each one them includes several steps of different operations that are executed on the events from
        the model server.
        Each event has metadata (function_uri, timestamp, class, etc.) but also inputs, predictions and optional
        metrics from the model server.
        In ths first part, the serving graph processes the event and splits it into sub-events. This part also includes
        validation of the event data and adding important details to the event such as endpoint_id.
        In the next parts, the serving graph stores data to 3 different targets:
        1. KV/SQL: Metadata and basic stats about the average latency and the amount of predictions over
           time per endpoint. for example the amount of predictions of endpoint x in the last 5 min. The model
           endpoints table also contains data on the model endpoint from other processes, such as feature_stats that
           represents sample statistics from the training data. If the target is from type KV, then the model endpoints
           table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/. If the target is
           SQL, then the table is stored within the database that was defined in the provided connection string.
        2. TSDB: live data of different key metric dictionaries in tsdb target.
           This data is being used by the monitoring dashboards in grafana. If using V3IO TSDB, results
           can be found under  v3io:///users/pipelines/project-name/model-endpoints/events/. In that case, we generate
           3 different key  metric dictionaries: base_metrics (average latency and predictions over time),
           endpoint_features (Prediction and feature names and values), and custom_metrics (user-defined metrics).
        3. Parquet: This Parquet file includes the required data for the model monitoring applications. If defined,
           the parquet target path can be found under mlrun.mlconf.model_endpoint_monitoring.offline. Otherwise,
           the default parquet path is under mlrun.mlconf.model_endpoint_monitoring.user_space. Note that if you are
           using CE, the parquet target path is based on the defined MLRun artifact path.

        :param fn: A serving function.
        :param tsdb_connector: Time series database connector.
        :param endpoint_store: KV/SQL store used for endpoint data.
        """

        graph = typing.cast(
            mlrun.serving.states.RootFlowStep,
            fn.set_topology(mlrun.serving.states.StepKinds.flow),
        )
        graph.add_step(
            "ExtractEndpointID",
            "extract_endpoint",
            full_event=True,
        )

        # split the graph between event with error vs valid event
        graph.add_step(
            "storey.Filter",
            "FilterError",
            after="extract_endpoint",
            _fn="(event.get('error') is None)",
        )

        graph.add_step(
            "storey.Filter",
            "ForwardError",
            after="extract_endpoint",
            _fn="(event.get('error') is not None)",
        )

        tsdb_connector.handle_model_error(
            graph,
        )

        # Process endpoint event: splitting into sub-events and validate event data
        def apply_process_endpoint_event():
            graph.add_step(
                "ProcessEndpointEvent",
                after="extract_endpoint",  # TODO: change this to FilterError in ML-7456
                full_event=True,
                project=self.project,
            )

        apply_process_endpoint_event()

        # Applying Storey operations of filtering and flatten
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

        # Validating feature names and map each feature to its value
        def apply_map_feature_names():
            graph.add_step(
                "MapFeatureNames",
                name="MapFeatureNames",
                infer_columns_from_data=True,
                project=self.project,
                after="flatten_events",
            )

        apply_map_feature_names()

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
                        "windows": self.aggregate_windows,
                        "period": self.aggregate_period,
                    }
                ],
                name=EventFieldType.LATENCY,
                after="MapFeatureNames",
                step_name="Aggregates",
                table=".",
                key_field=EventFieldType.ENDPOINT_ID,
            )
            # Calculate average latency time for each window (5 min and 1 hour by default)
            graph.add_step(
                class_name="storey.Rename",
                mapping={
                    "latency_count_5m": EventLiveStats.PREDICTIONS_COUNT_5M,
                    "latency_count_1h": EventLiveStats.PREDICTIONS_COUNT_1H,
                },
                name="Rename",
                after=EventFieldType.LATENCY,
            )

        apply_storey_aggregations()

        # KV/SQL branch
        # Filter relevant keys from the event before writing the data into the database table
        def apply_process_before_endpoint_update():
            graph.add_step(
                "ProcessBeforeEndpointUpdate",
                name="ProcessBeforeEndpointUpdate",
                after="Rename",
            )

        apply_process_before_endpoint_update()

        # Write the filtered event to KV/SQL table. At this point, the serving graph updates the stats
        # about average latency and the amount of predictions over time
        def apply_update_endpoint():
            graph.add_step(
                "UpdateEndpoint",
                name="UpdateEndpoint",
                after="ProcessBeforeEndpointUpdate",
                project=self.project,
            )

        apply_update_endpoint()

        # (only for V3IO KV target) - Apply infer_schema on the model endpoints table for generating schema file
        # which will be used by Grafana monitoring dashboards
        def apply_infer_schema():
            graph.add_step(
                "InferSchema",
                name="InferSchema",
                after="UpdateEndpoint",
                v3io_framesd=self.v3io_framesd,
                container=self.kv_container,
                table=self.kv_path,
            )

        if endpoint_store.type == ModelEndpointTarget.V3IO_NOSQL:
            apply_infer_schema()

        tsdb_connector.apply_monitoring_stream_steps(graph=graph)

        # Parquet branch
        # Filter and validate different keys before writing the data to Parquet target
        def apply_process_before_parquet():
            graph.add_step(
                "ProcessBeforeParquet",
                name="ProcessBeforeParquet",
                after="MapFeatureNames",
                _fn="(event)",
            )

        apply_process_before_parquet()

        # Write the Parquet target file, partitioned by key (endpoint_id) and time.
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
                time_field=EventFieldType.TIMESTAMP,
                partition_cols=["$key", "$year", "$month", "$day", "$hour"],
            )

        apply_parquet_target()


class ProcessBeforeEndpointUpdate(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Filter relevant keys from the event before writing the data to database table (in EndpointUpdate step).
        Note that in the endpoint table we only keep metadata (function_uri, model_class, etc.) and stats about the
        average latency and the number of predictions (per 5min and 1hour).

        :returns: A filtered event as a dictionary which will be written to the endpoint table in the next step.
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
                EventFieldType.ENDPOINT_ID,
                EventFieldType.LABELS,
                EventFieldType.FIRST_REQUEST,
                EventFieldType.LAST_REQUEST,
                EventFieldType.ERROR_COUNT,
            ]
        }

        # Add generic metrics statistics
        generic_metrics = {
            k: event[k]
            for k in [
                EventLiveStats.LATENCY_AVG_5M,
                EventLiveStats.LATENCY_AVG_1H,
                EventLiveStats.PREDICTIONS_PER_SECOND,
                EventLiveStats.PREDICTIONS_COUNT_5M,
                EventLiveStats.PREDICTIONS_COUNT_1H,
            ]
        }

        e[EventFieldType.METRICS] = json.dumps(
            {EventKeyMetrics.GENERIC: generic_metrics}
        )

        # Write labels as json string as required by the DB format
        e[EventFieldType.LABELS] = json.dumps(e[EventFieldType.LABELS])

        return e


class ExtractEndpointID(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs) -> None:
        """
        Generate the model endpoint ID based on the event parameters and attach it to the event.
        """
        super().__init__(**kwargs)

    def do(self, full_event) -> typing.Union[storey.Event, None]:
        # Getting model version and function uri from event
        # and use them for retrieving the endpoint_id
        function_uri = full_event.body.get(EventFieldType.FUNCTION_URI)
        if not is_not_none(function_uri, [EventFieldType.FUNCTION_URI]):
            return None

        model = full_event.body.get(EventFieldType.MODEL)
        if not is_not_none(model, [EventFieldType.MODEL]):
            return None

        version = full_event.body.get(EventFieldType.VERSION)
        versioned_model = f"{model}:{version}" if version else f"{model}:latest"

        endpoint_id = mlrun.common.model_monitoring.create_model_endpoint_uid(
            function_uri=function_uri,
            versioned_model=versioned_model,
        )

        endpoint_id = str(endpoint_id)
        full_event.body[EventFieldType.ENDPOINT_ID] = endpoint_id
        full_event.body[EventFieldType.VERSIONED_MODEL] = versioned_model
        return full_event


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
            EventFieldType.FEATURES,
            EventFieldType.NAMED_FEATURES,
            EventFieldType.PREDICTION,
            EventFieldType.NAMED_PREDICTIONS,
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
        project: str,
        **kwargs,
    ):
        """
        Process event or batch of events as part of the first step of the monitoring serving graph. It includes
        Adding important details to the event such as endpoint_id, handling errors coming from the stream, validation
        of event data such as inputs and outputs, and splitting model event into sub-events.

        :param project: Project name.

        :returns: A Storey event object which is the basic unit of data in Storey. Note that the next steps of
                  the monitoring serving graph are based on Storey operations.

        """
        super().__init__(**kwargs)

        self.project: str = project

        # First and last requests timestamps (value) of each endpoint (key)
        self.first_request: dict[str, str] = dict()
        self.last_request: dict[str, str] = dict()

        # Number of errors (value) per endpoint (key)
        self.error_count: dict[str, int] = collections.defaultdict(int)

        # Set of endpoints in the current events
        self.endpoints: set[str] = set()

    def do(self, full_event):
        event = full_event.body

        versioned_model = event[EventFieldType.VERSIONED_MODEL]
        endpoint_id = event[EventFieldType.ENDPOINT_ID]
        function_uri = event[EventFieldType.FUNCTION_URI]

        # In case this process fails, resume state from existing record
        self.resume_state(endpoint_id)

        # If error key has been found in the current event,
        # increase the error counter by 1 and raise the error description
        error = event.get("error")
        if error:  # TODO: delete this in ML-7456
            self.error_count[endpoint_id] += 1
            raise mlrun.errors.MLRunInvalidArgumentError(str(error))

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
            # Set time for the first request of the current endpoint
            self.first_request[endpoint_id] = timestamp

        # Validate that the request time of the current event is later than the previous request time
        self._validate_last_request_timestamp(
            endpoint_id=endpoint_id, timestamp=timestamp
        )

        # Set time for the last reqeust of the current endpoint
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

        # Convert timestamp to a datetime object
        timestamp = datetime.datetime.fromisoformat(timestamp)

        # Separate each model invocation into sub events that will be stored as dictionary
        # in list of events. This list will be used as the body for the storey event.
        if not isinstance(features, list):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Model's inputs must be a list"
            )
        features = (
            features
            if not any(not isinstance(feat, list) for feat in features)
            else [features]
        )
        if not isinstance(predictions, list):
            predictions = [[predictions]]
        elif isinstance(predictions, list) and len(predictions) == len(features):
            pass  # predictions are already in the right format
        else:
            predictions = (
                predictions
                if not any(not isinstance(pred, list) for pred in predictions)
                else [predictions]
            )

        events = []
        for i, (feature, prediction) in enumerate(zip(features, predictions)):
            if not isinstance(prediction, list):
                prediction = [prediction]

            if not isinstance(feature, list):
                feature = [feature]

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
                    EventFieldType.LAST_REQUEST_TIMESTAMP: mlrun.utils.enrich_datetime_with_tz_info(
                        self.last_request[endpoint_id]
                    ).timestamp(),
                    EventFieldType.ERROR_COUNT: self.error_count[endpoint_id],
                    EventFieldType.LABELS: event.get(EventFieldType.LABELS, {}),
                    EventFieldType.METRICS: event.get(EventFieldType.METRICS, {}),
                    EventFieldType.ENTITIES: event.get("request", {}).get(
                        EventFieldType.ENTITIES, {}
                    ),
                }
            )

        # Create a storey event object with list of events, based on endpoint_id which will be used
        # in the upcoming steps
        storey_event = storey.Event(body=events, key=endpoint_id)
        return storey_event

    def _validate_last_request_timestamp(self, endpoint_id: str, timestamp: str):
        """Validate that the request time of the current event is later than the previous request time that has
        already been processed.

        :param endpoint_id: The unique id of the model endpoint.
        :param timestamp:   Event request time as a string.

        :raise MLRunPreconditionFailedError: If the request time of the current is later than the previous request time.
        """

        if (
            endpoint_id in self.last_request
            and self.last_request[endpoint_id] > timestamp
        ):
            logger.error(
                f"current event request time {timestamp} is earlier than the last request time "
                f"{self.last_request[endpoint_id]} - write to TSDB will be rejected"
            )

    def resume_state(self, endpoint_id):
        # Make sure process is resumable, if process fails for any reason, be able to pick things up close to where we
        # left them
        if endpoint_id not in self.endpoints:
            logger.info("Trying to resume state", endpoint_id=endpoint_id)
            endpoint_record = mlrun.model_monitoring.helpers.get_endpoint_record(
                project=self.project,
                endpoint_id=endpoint_id,
            )

            # If model endpoint found, get first_request, last_request and error_count values
            if endpoint_record:
                first_request = endpoint_record.get(EventFieldType.FIRST_REQUEST)

                if first_request:
                    self.first_request[endpoint_id] = first_request

                last_request = endpoint_record.get(EventFieldType.LAST_REQUEST)
                if last_request:
                    self.last_request[endpoint_id] = last_request

                error_count = endpoint_record.get(EventFieldType.ERROR_COUNT)

                if error_count:
                    self.error_count[endpoint_id] = int(error_count)

            # add endpoint to endpoints set
            self.endpoints.add(endpoint_id)

    def is_valid(
        self,
        endpoint_id: str,
        validation_function,
        field: typing.Any,
        dict_path: list[str],
    ):
        if validation_function(field, dict_path):
            return True
        self.error_count[endpoint_id] += 1
        return False


def is_not_none(field: typing.Any, dict_path: list[str]):
    if field is not None:
        return True
    logger.error(
        f"Expected event field is missing: {field} [Event -> {','.join(dict_path)}]"
    )
    return False


class MapFeatureNames(mlrun.feature_store.steps.MapClass):
    def __init__(
        self,
        project: str,
        infer_columns_from_data: bool = False,
        **kwargs,
    ):
        """
        Validating feature names and label columns and map each feature to its value. In the end of this step,
        the event should have key-value pairs of (feature name: feature value).

        :param project:                 Project name.
        :param infer_columns_from_data: If true and features or labels names were not found, then try to
                                        retrieve them from data that was stored in the previous events of
                                        the current process. This data can be found under self.feature_names and
                                        self.label_columns.


        :returns: A single event as a dictionary that includes metadata (endpoint_id, model_class, etc.) and also
                  feature names and values (as well as the prediction results).
        """
        super().__init__(**kwargs)

        self._infer_columns_from_data = infer_columns_from_data
        self.project = project

        # Dictionaries that will be used in case features names
        # and labels columns were not found in the current event
        self.feature_names = {}
        self.label_columns = {}

        # Dictionary to manage the model endpoint types - important for the V3IO TSDB
        self.endpoint_type = {}

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

    def do(self, event: dict):
        endpoint_id = event[EventFieldType.ENDPOINT_ID]

        feature_values = event[EventFieldType.FEATURES]
        label_values = event[EventFieldType.PREDICTION]

        for index in range(len(feature_values)):
            feature_value = feature_values[index]
            if isinstance(feature_value, int):
                feature_values[index] = float(feature_value)

        # Get feature names and label columns
        if endpoint_id not in self.feature_names:
            endpoint_record = mlrun.model_monitoring.helpers.get_endpoint_record(
                project=self.project,
                endpoint_id=endpoint_id,
            )
            feature_names = endpoint_record.get(EventFieldType.FEATURE_NAMES)
            feature_names = json.loads(feature_names) if feature_names else None

            label_columns = endpoint_record.get(EventFieldType.LABEL_NAMES)
            label_columns = json.loads(label_columns) if label_columns else None

            # If feature names were not found,
            # try to retrieve them from the previous events of the current process
            if not feature_names and self._infer_columns_from_data:
                feature_names = self._infer_feature_names_from_data(event)

            endpoint_type = int(endpoint_record.get(EventFieldType.ENDPOINT_TYPE))
            if not feature_names:
                logger.warn(
                    "Feature names are not initialized, they will be automatically generated",
                    endpoint_id=endpoint_id,
                )
                feature_names = [
                    f"f{i}" for i, _ in enumerate(event[EventFieldType.FEATURES])
                ]

                # Update the endpoint record with the generated features
                update_endpoint_record(
                    project=self.project,
                    endpoint_id=endpoint_id,
                    attributes={
                        EventFieldType.FEATURE_NAMES: json.dumps(feature_names)
                    },
                )

                if endpoint_type != EndpointType.ROUTER.value:
                    update_monitoring_feature_set(
                        endpoint_record=endpoint_record,
                        feature_names=feature_names,
                        feature_values=feature_values,
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

                update_endpoint_record(
                    project=self.project,
                    endpoint_id=endpoint_id,
                    attributes={EventFieldType.LABEL_NAMES: json.dumps(label_columns)},
                )
                if endpoint_type != EndpointType.ROUTER.value:
                    update_monitoring_feature_set(
                        endpoint_record=endpoint_record,
                        feature_names=label_columns,
                        feature_values=label_values,
                    )

            self.label_columns[endpoint_id] = label_columns
            self.feature_names[endpoint_id] = feature_names

            logger.info(
                "Label columns", endpoint_id=endpoint_id, label_columns=label_columns
            )
            logger.info(
                "Feature names", endpoint_id=endpoint_id, feature_names=feature_names
            )

            # Update the endpoint type within the endpoint types dictionary
            self.endpoint_type[endpoint_id] = endpoint_type

        # Add feature_name:value pairs along with a mapping dictionary of all of these pairs
        feature_names = self.feature_names[endpoint_id]
        self._map_dictionary_values(
            event=event,
            named_iters=feature_names,
            values_iters=feature_values,
            mapping_dictionary=EventFieldType.NAMED_FEATURES,
        )

        # Add label_name:value pairs along with a mapping dictionary of all of these pairs
        label_names = self.label_columns[endpoint_id]
        self._map_dictionary_values(
            event=event,
            named_iters=label_names,
            values_iters=label_values,
            mapping_dictionary=EventFieldType.NAMED_PREDICTIONS,
        )

        # Add endpoint type to the event
        event[EventFieldType.ENDPOINT_TYPE] = self.endpoint_type[endpoint_id]

        logger.info("Mapped event", event=event)
        return event

    @staticmethod
    def _map_dictionary_values(
        event: dict,
        named_iters: list,
        values_iters: list,
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


class UpdateEndpoint(mlrun.feature_store.steps.MapClass):
    def __init__(self, project: str, **kwargs):
        """
        Update the model endpoint record in the DB. Note that the event at this point includes metadata and stats about
        the average latency and the amount of predictions over time. This data will be used in the monitoring dashboards
        such as "Model Monitoring - Performance" which can be found in Grafana.

        :returns: Event as a dictionary (without any changes) for the next step (InferSchema).
        """
        super().__init__(**kwargs)
        self.project = project

    def do(self, event: dict):
        # Remove labels from the event
        event.pop(EventFieldType.LABELS)

        update_endpoint_record(
            project=self.project,
            endpoint_id=event.pop(EventFieldType.ENDPOINT_ID),
            attributes=event,
        )
        return event


class InferSchema(mlrun.feature_store.steps.MapClass):
    def __init__(
        self,
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
        self.v3io_framesd = v3io_framesd
        self.table = table
        self.keys = set()

    def do(self, event: dict):
        key_set = set(event.keys())
        if not key_set.issubset(self.keys):
            import mlrun.utils.v3io_clients

            self.keys.update(key_set)
            # Apply infer_schema on the kv table for generating the schema file
            mlrun.utils.v3io_clients.get_frames_client(
                container=self.container,
                address=self.v3io_framesd,
            ).execute(backend="kv", table=self.table, command="infer_schema")

        return event


def update_endpoint_record(
    project: str,
    endpoint_id: str,
    attributes: dict,
):
    model_endpoint_store = mlrun.model_monitoring.get_store_object(
        project=project,
    )

    model_endpoint_store.update_model_endpoint(
        endpoint_id=endpoint_id, attributes=attributes
    )


def update_monitoring_feature_set(
    endpoint_record: dict[str, typing.Any],
    feature_names: list[str],
    feature_values: list[typing.Any],
):
    monitoring_feature_set = fstore.get_feature_set(
        endpoint_record[
            mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_SET_URI
        ]
    )
    for name, val in zip(feature_names, feature_values):
        monitoring_feature_set.add_feature(
            fstore.Feature(name=name, value_type=type(val))
        )

    monitoring_feature_set.save()
