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

import datetime
import typing

import mlrun
import mlrun.common.model_monitoring.helpers
import mlrun.feature_store as fstore
import mlrun.feature_store.steps
import mlrun.serving.states
import mlrun.utils
from mlrun.common.schemas.model_monitoring.constants import (
    ControllerEvent,
    ControllerEventKind,
    EndpointType,
    EventFieldType,
    FileTargetKind,
    ProjectSecretKeys,
)
from mlrun.model_monitoring.db import TSDBConnector
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

        self.tsdb_configurations = {}
        if not mlrun.mlconf.is_ce_mode():
            self._initialize_v3io_configurations(
                model_monitoring_access_key=model_monitoring_access_key
            )

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

        self.v3io_access_key = v3io_access_key or mlrun.mlconf.get_v3io_access_key()
        self.model_monitoring_access_key = (
            model_monitoring_access_key
            or mlrun.get_secret_or_env(ProjectSecretKeys.ACCESS_KEY)
            or self.v3io_access_key
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
        controller_stream_uri: str,
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
        :param controller_stream_uri: The controller stream URI. Runs on server api pod so needed to be provided as
        input
        """

        graph = typing.cast(
            mlrun.serving.states.RootFlowStep,
            fn.set_topology(mlrun.serving.states.StepKinds.flow, engine="async"),
        )

        # split the graph between event with error vs valid event
        graph.add_step(
            "storey.Filter",
            "FilterError",
            _fn="(event.get('error') is None)",
        )

        graph.add_step(
            "storey.Filter",
            "ForwardError",
            _fn="(event.get('error') is not None)",
        )

        tsdb_connector.handle_model_error(
            graph,
        )

        # Process endpoint event: splitting into sub-events and validate event data
        def apply_process_endpoint_event():
            graph.add_step(
                "ProcessEndpointEvent",
                after="FilterError",
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
        # split the graph between event with error vs valid event
        graph.add_step(
            "storey.Filter",
            "FilterNOP",
            after="MapFeatureNames",
            _fn="(event.get('kind', " ") != 'nop_event')",
        )
        graph.add_step(
            "storey.Filter",
            "ForwardNOP",
            after="MapFeatureNames",
            _fn="(event.get('kind', " ") == 'nop_event')",
        )

        tsdb_connector.apply_monitoring_stream_steps(
            graph=graph,
            aggregate_windows=self.aggregate_windows,
            aggregate_period=self.aggregate_period,
        )

        # Parquet branch
        # Filter and validate different keys before writing the data to Parquet target
        def apply_process_before_parquet():
            graph.add_step(
                "ProcessBeforeParquet",
                name="ProcessBeforeParquet",
                after="FilterNOP",
                _fn="(event)",
            )

        apply_process_before_parquet()

        # Write the Parquet target file, partitioned by key (endpoint_id) and time.
        def apply_parquet_target():
            graph.add_step(
                "mlrun.datastore.storeytargets.ParquetStoreyTarget",
                alternative_v3io_access_key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
                name="ParquetTarget",
                after="ProcessBeforeParquet",
                graph_shape="cylinder",
                path=self.parquet_path,
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

        # controller branch
        def apply_push_controller_stream(stream_uri: str):
            graph.add_step(
                ">>",
                "controller_stream",
                path=stream_uri,
                sharding_func=ControllerEvent.ENDPOINT_ID,
                after="ForwardNOP",
                # Force using the pipeline key instead of the one in the profile in case of v3io profile.
                # In case of Kafka, this parameter will be ignored.
                alternative_v3io_access_key="V3IO_ACCESS_KEY",
            )

        apply_push_controller_stream(controller_stream_uri)


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

        # Set of endpoints in the current events
        self.endpoints: set[str] = set()

    def do(self, full_event):
        event = full_event.body
        if event.get(ControllerEvent.KIND, "") == ControllerEventKind.NOP_EVENT:
            logger.debug(
                "Skipped nop event inside of ProcessEndpointEvent", event=event
            )
            full_event.body = [event]
            return full_event
        # Getting model version and function uri from event
        # and use them for retrieving the endpoint_id
        function_uri = full_event.body.get(EventFieldType.FUNCTION_URI)
        if not is_not_none(function_uri, [EventFieldType.FUNCTION_URI]):
            return None

        model = full_event.body.get(EventFieldType.MODEL)
        if not is_not_none(model, [EventFieldType.MODEL]):
            return None

        endpoint_id = event[EventFieldType.ENDPOINT_ID]

        # In case this process fails, resume state from existing record
        self.resume_state(
            endpoint_id=endpoint_id,
            endpoint_name=full_event.body.get(EventFieldType.MODEL),
        )

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
            validation_function=is_not_none,
            field=timestamp,
            dict_path=["when"],
        ):
            return None

        if endpoint_id not in self.first_request:
            # Set time for the first request of the current endpoint
            self.first_request[endpoint_id] = timestamp

        if not self.is_valid(
            validation_function=is_not_none,
            field=request_id,
            dict_path=["request", "id"],
        ):
            return None
        if not self.is_valid(
            validation_function=is_not_none,
            field=latency,
            dict_path=["microsec"],
        ):
            return None
        if not self.is_valid(
            validation_function=is_not_none,
            field=features,
            dict_path=["request", "inputs"],
        ):
            return None
        if not self.is_valid(
            validation_function=is_not_none,
            field=predictions,
            dict_path=["resp", "outputs"],
        ):
            return None

        # Convert timestamp to a datetime object
        timestamp_obj = datetime.datetime.fromisoformat(timestamp)

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

            effective_sample_count, estimated_prediction_count = (
                self._get_effective_and_estimated_counts(event=event)
            )

            events.append(
                {
                    EventFieldType.FUNCTION_URI: function_uri,
                    EventFieldType.ENDPOINT_NAME: event.get(EventFieldType.MODEL),
                    EventFieldType.MODEL_CLASS: model_class,
                    EventFieldType.TIMESTAMP: timestamp_obj,
                    EventFieldType.ENDPOINT_ID: endpoint_id,
                    EventFieldType.REQUEST_ID: request_id,
                    EventFieldType.LATENCY: latency,
                    EventFieldType.FEATURES: feature,
                    EventFieldType.PREDICTION: prediction,
                    EventFieldType.FIRST_REQUEST: self.first_request[endpoint_id],
                    EventFieldType.LAST_REQUEST: timestamp,
                    EventFieldType.LAST_REQUEST_TIMESTAMP: mlrun.utils.enrich_datetime_with_tz_info(
                        timestamp
                    ).timestamp(),
                    EventFieldType.LABELS: event.get(EventFieldType.LABELS, {}),
                    EventFieldType.METRICS: event.get(EventFieldType.METRICS, {}),
                    EventFieldType.ENTITIES: event.get("request", {}).get(
                        EventFieldType.ENTITIES, {}
                    ),
                    EventFieldType.EFFECTIVE_SAMPLE_COUNT: effective_sample_count,
                    EventFieldType.ESTIMATED_PREDICTION_COUNT: estimated_prediction_count,
                }
            )

        # Create a storey event object with list of events, based on endpoint_id which will be used
        # in the upcoming steps
        full_event.key = endpoint_id
        full_event.body = events
        return full_event

    def resume_state(self, endpoint_id, endpoint_name):
        # Make sure process is resumable, if process fails for any reason, be able to pick things up close to where we
        # left them
        if endpoint_id not in self.endpoints:
            logger.info("Trying to resume state", endpoint_id=endpoint_id)
            endpoint_record = (
                mlrun.db.get_run_db()
                .get_model_endpoint(
                    project=self.project,
                    endpoint_id=endpoint_id,
                    name=endpoint_name,
                    tsdb_metrics=False,
                )
                .flat_dict()
            )

            # If model endpoint found, get first_request & last_request values
            if endpoint_record:
                first_request = endpoint_record.get(EventFieldType.FIRST_REQUEST)

                if first_request:
                    self.first_request[endpoint_id] = first_request

            # add endpoint to endpoints set
            self.endpoints.add(endpoint_id)

    def is_valid(
        self,
        validation_function,
        field: typing.Any,
        dict_path: list[str],
    ):
        if validation_function(field, dict_path):
            return True

        return False

    @staticmethod
    def _get_effective_and_estimated_counts(event):
        """
        Calculate the `effective_sample_count` and the `estimated_prediction_count` based on the event's
        sampling percentage. These values will be stored in the TSDB target.
        Note that In non-batch serving, the `effective_sample_count` is always set to 1. In addition, when the sampling
        percentage is 100%, the `estimated_prediction_count` is equal to the `effective_sample_count`.
        """
        effective_sample_count = event.get(EventFieldType.EFFECTIVE_SAMPLE_COUNT, 1)
        estimated_prediction_count = effective_sample_count * (
            100 / event.get(EventFieldType.SAMPLING_PERCENTAGE, 100)
        )
        return effective_sample_count, estimated_prediction_count


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
        self.first_request = {}

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
        if event.get(ControllerEvent.KIND, "") == ControllerEventKind.NOP_EVENT:
            return event
        endpoint_id = event[EventFieldType.ENDPOINT_ID]

        feature_values = event[EventFieldType.FEATURES]
        label_values = event[EventFieldType.PREDICTION]

        for index in range(len(feature_values)):
            feature_value = feature_values[index]
            if isinstance(feature_value, int):
                feature_values[index] = float(feature_value)

        attributes_to_update = {}
        endpoint_record = None
        # Get feature names and label columns
        if endpoint_id not in self.feature_names:
            endpoint_record = (
                mlrun.db.get_run_db()
                .get_model_endpoint(
                    project=self.project,
                    endpoint_id=endpoint_id,
                    name=event[EventFieldType.ENDPOINT_NAME],
                    tsdb_metrics=False,
                )
                .flat_dict()
            )
            feature_names = endpoint_record.get(EventFieldType.FEATURE_NAMES)

            label_columns = endpoint_record.get(EventFieldType.LABEL_NAMES)

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
                attributes_to_update[EventFieldType.FEATURE_NAMES] = feature_names

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
                attributes_to_update[EventFieldType.LABEL_NAMES] = label_columns
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

        # Update the first request time in the endpoint record
        if endpoint_id not in self.first_request:
            endpoint_record = endpoint_record or (
                mlrun.db.get_run_db()
                .get_model_endpoint(
                    project=self.project,
                    endpoint_id=endpoint_id,
                    name=event[EventFieldType.ENDPOINT_NAME],
                    tsdb_metrics=False,
                )
                .flat_dict()
            )
            if not endpoint_record.get(EventFieldType.FIRST_REQUEST):
                attributes_to_update[EventFieldType.FIRST_REQUEST] = (
                    mlrun.utils.enrich_datetime_with_tz_info(
                        event[EventFieldType.FIRST_REQUEST]
                    )
                )
            self.first_request[endpoint_id] = True

        if attributes_to_update:
            logger.info(
                "Updating endpoint record",
                endpoint_id=endpoint_id,
                attributes=attributes_to_update,
            )
            update_endpoint_record(
                project=self.project,
                endpoint_id=endpoint_id,
                attributes=attributes_to_update,
                endpoint_name=event[EventFieldType.ENDPOINT_NAME],
            )

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
        diff = len(named_iters) - len(values_iters)
        values_iters += [None] * diff
        for name, value in zip(named_iters, values_iters):
            event[name] = value
            event[mapping_dictionary][name] = value


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
    endpoint_name: str,
    attributes: dict,
):
    mlrun.db.get_run_db().patch_model_endpoint(
        project=project,
        endpoint_id=endpoint_id,
        attributes=attributes,
        name=endpoint_name,
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
