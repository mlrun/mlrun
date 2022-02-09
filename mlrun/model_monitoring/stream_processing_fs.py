import json
import os
from collections import defaultdict
from os import environ
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import v3io

# Constants
from storey import Event
from v3io.dataplane import RaiseForStatus

import mlrun.feature_store as fs
from mlrun.config import config
from mlrun.datastore.targets import ParquetTarget
from mlrun.feature_store.steps import MapClass
from mlrun.utils import logger
from mlrun.utils.model_monitoring import (
    create_model_endpoint_id,
    parse_model_endpoint_store_prefix,
)
from mlrun.utils.v3io_clients import get_frames_client, get_v3io_client

ISO_8061_UTC = "%Y-%m-%d %H:%M:%S.%f%z"
FUNCTION_URI = "function_uri"
MODEL = "model"
VERSION = "version"
VERSIONED_MODEL = "versioned_model"
MODEL_CLASS = "model_class"
TIMESTAMP = "timestamp"
ENDPOINT_ID = "endpoint_id"
REQUEST_ID = "request_id"
LABELS = "labels"
UNPACKED_LABELS = "unpacked_labels"
LATENCY_AVG_5M = "latency_avg_5m"
LATENCY_AVG_1H = "latency_avg_1h"
PREDICTIONS_PER_SECOND = "predictions_per_second"
PREDICTIONS_COUNT_5M = "predictions_count_5m"
PREDICTIONS_COUNT_1H = "predictions_count_1h"
FIRST_REQUEST = "first_request"
LAST_REQUEST = "last_request"
ERROR_COUNT = "error_count"
ENTITIES = "entities"
FEATURE_NAMES = "feature_names"
LABEL_COLUMNS = "label_columns"
LATENCY = "latency"
RECORD_TYPE = "record_type"
FEATURES = "features"
PREDICTION = "prediction"
PREDICTIONS = "predictions"
NAMED_FEATURES = "named_features"
NAMED_PREDICTIONS = "named_predictions"
BASE_METRICS = "base_metrics"
CUSTOM_METRICS = "custom_metrics"
ENDPOINT_FEATURES = "endpoint_features"
METRICS = "metrics"
BATCH_TIMESTAMP = "batch_timestamp"
TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"  # ISO 8061


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
        aggregate_count_windows: Optional[List[str]] = None,
        aggregate_count_period: str = "30s",
        aggregate_avg_windows: Optional[List[str]] = None,
        aggregate_avg_period: str = "30s",
        v3io_access_key: Optional[str] = None,
        v3io_framesd: Optional[str] = None,
        v3io_api: Optional[str] = None,
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

        self.v3io_framesd = v3io_framesd or config.v3io_framesd
        self.v3io_api = v3io_api or config.v3io_api

        self.v3io_access_key = v3io_access_key or environ.get("V3IO_ACCESS_KEY")
        self.model_monitoring_access_key = (
            model_monitoring_access_key
            or os.environ.get("MODEL_MONITORING_ACCESS_KEY")
            or self.v3io_access_key
        )

        template = config.model_endpoint_monitoring.store_prefixes.default

        kv_path = template.format(project=project, kind="endpoints")
        _, self.kv_container, self.kv_path = parse_model_endpoint_store_prefix(kv_path)

        tsdb_path = template.format(project=project, kind="events")
        _, self.tsdb_container, self.tsdb_path = parse_model_endpoint_store_prefix(
            tsdb_path
        )
        self.tsdb_path = f"{self.tsdb_container}/{self.tsdb_path}"

        self.parquet_path = config.model_endpoint_monitoring.store_prefixes.user_space.format(
            project=project, kind="parquet"
        )

        logger.info(
            "Initializing model monitoring event stream processor",
            parquet_batching_max_events=self.parquet_batching_max_events,
            v3io_access_key=self.v3io_access_key,
            model_monitoring_access_key=self.model_monitoring_access_key,
            default_store_prefix=config.model_endpoint_monitoring.store_prefixes.default,
            user_space_store_prefix=config.model_endpoint_monitoring.store_prefixes.user_space,
            v3io_api=self.v3io_api,
            v3io_framesd=self.v3io_framesd,
            kv_container=self.kv_container,
            kv_path=self.kv_path,
            tsdb_container=self.tsdb_container,
            tsdb_path=self.tsdb_path,
            parquet_path=self.parquet_path,
        )

    def create_feature_set(self):
        feature_set = fs.FeatureSet(
            "monitoring", entities=[ENDPOINT_ID], timestamp_key=TIMESTAMP
        )
        feature_set.graph.to(
            "ProcessEndpointEvent",
            kv_container=self.kv_container,
            kv_path=self.kv_path,
            v3io_access_key=self.v3io_access_key,
            full_event=True,
        ).to("storey.Filter", "filter_none", _fn="(event is not None)").to(
            "storey.FlatMap", "flatten_events", _fn="(event)"
        ).to(
            "MapFeatureNames",
            name="MapFeatureNames",
            kv_container=self.kv_container,
            kv_path=self.kv_path,
            access_key=self.v3io_access_key,
            infer_columns_from_data=True,
        )
        # kv and tsdb branch
        feature_set.add_aggregation(
            ENDPOINT_ID,
            ["count"],
            self.aggregate_count_windows,
            self.aggregate_count_period,
            name=PREDICTIONS,
            after="MapFeatureNames",
            step_name="Aggregates",
        )
        feature_set.add_aggregation(
            LATENCY, ["avg"], self.aggregate_avg_windows, self.aggregate_avg_period,
        )
        feature_set.graph.add_step(
            "storey.steps.SampleWindow",
            name="sample",
            after="Aggregates",
            window_size=self.sample_window,
            key=ENDPOINT_ID,
        )
        # kv
        feature_set.graph.add_step(
            "ProcessBeforeKV", name="ProcessBeforeKV", after="sample"
        )
        feature_set.graph.add_step(
            "WriteToKV",
            name="WriteToKV",
            after="ProcessBeforeKV",
            container=self.kv_container,
            table=self.kv_path,
        )
        feature_set.graph.add_step(
            "InferSchema",
            name="InferSchema",
            after="WriteToKV",
            v3io_access_key=self.v3io_access_key,
            v3io_framesd=self.v3io_framesd,
            container=self.kv_container,
            table=self.kv_path,
        )
        # tsdb
        feature_set.graph.add_step(
            "ProcessBeforeTSDB", name="ProcessBeforeTSDB", after="sample"
        )
        feature_set.graph.add_step(
            "FilterAndUnpackKeys",
            name="FilterAndUnpackKeys1",
            after="ProcessBeforeTSDB",
            keys=[BASE_METRICS],
        )
        feature_set.graph.add_step(
            "storey.TSDBTarget",
            name="tsdb1",
            after="FilterAndUnpackKeys1",
            path=self.tsdb_path,
            rate="10/m",
            time_col=TIMESTAMP,
            container=self.tsdb_container,
            access_key=self.v3io_access_key,
            v3io_frames=self.v3io_framesd,
            index_cols=[ENDPOINT_ID, RECORD_TYPE],
            max_events=self.tsdb_batching_max_events,
            timeout_secs=self.tsdb_batching_timeout_secs,
            key=ENDPOINT_ID,
        )
        feature_set.graph.add_step(
            "FilterAndUnpackKeys",
            name="FilterAndUnpackKeys2",
            after="ProcessBeforeTSDB",
            keys=[ENDPOINT_FEATURES],
        )
        feature_set.graph.add_step(
            "storey.TSDBTarget",
            name="tsdb2",
            after="FilterAndUnpackKeys2",
            path=self.tsdb_path,
            rate="10/m",
            time_col=TIMESTAMP,
            container=self.tsdb_container,
            access_key=self.v3io_access_key,
            v3io_frames=self.v3io_framesd,
            index_cols=[ENDPOINT_ID, RECORD_TYPE],
            max_events=self.tsdb_batching_max_events,
            timeout_secs=self.tsdb_batching_timeout_secs,
            key=ENDPOINT_ID,
        )
        feature_set.graph.add_step(
            "FilterAndUnpackKeys",
            name="FilterAndUnpackKeys3",
            after="ProcessBeforeTSDB",
            keys=[CUSTOM_METRICS],
        )
        feature_set.graph.add_step(
            "storey.Filter",
            "FilterNotNone",
            after="FilterAndUnpackKeys3",
            _fn="(event is not None)",
        )
        feature_set.graph.add_step(
            "storey.TSDBTarget",
            name="tsdb3",
            after="FilterNotNone",
            path=self.tsdb_path,
            rate="10/m",
            time_col=TIMESTAMP,
            container=self.tsdb_container,
            access_key=self.v3io_access_key,
            v3io_frames=self.v3io_framesd,
            index_cols=[ENDPOINT_ID, RECORD_TYPE],
            max_events=self.tsdb_batching_max_events,
            timeout_secs=self.tsdb_batching_timeout_secs,
            key=ENDPOINT_ID,
        )

        # parquet branch
        feature_set.graph.add_step(
            "ProcessBeforeParquet",
            name="ProcessBeforeParquet",
            after="MapFeatureNames",
            _fn="(event)",
        )
        storage_options = dict(
            v3io_access_key=self.model_monitoring_access_key, v3io_api=self.v3io_api
        )

        pq_target = ParquetTarget(
            path=self.parquet_path,
            after_step="ProcessBeforeParquet",
            key_bucketing_number=0,
            time_partitioning_granularity="hour",
            max_events=self.parquet_batching_max_events,
            flush_after_seconds=self.parquet_batching_timeout_secs,
            storage_options=storage_options,
            attributes={"infer_columns_from_data": True},
        )

        feature_set.set_targets(
            targets=[pq_target],
            with_defaults=False,
            default_final_step="ProcessBeforeParquet",
        )
        return feature_set


class ProcessBeforeKV(MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, event):
        # compute prediction per second
        event[PREDICTIONS_PER_SECOND] = float(event[PREDICTIONS_COUNT_5M]) / 300
        # Filter relevant keys
        e = {
            k: event[k]
            for k in [
                FUNCTION_URI,
                MODEL,
                MODEL_CLASS,
                TIMESTAMP,
                ENDPOINT_ID,
                LABELS,
                UNPACKED_LABELS,
                LATENCY_AVG_5M,
                LATENCY_AVG_1H,
                PREDICTIONS_PER_SECOND,
                PREDICTIONS_COUNT_5M,
                PREDICTIONS_COUNT_1H,
                FIRST_REQUEST,
                LAST_REQUEST,
                ERROR_COUNT,
            ]
        }
        # Unpack labels dictionary
        e = {**e, **e.pop(UNPACKED_LABELS, {})}
        # Write labels to kv as json string to be presentable later
        e[LABELS] = json.dumps(e[LABELS])
        return e


class ProcessBeforeTSDB(MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, event):
        # compute prediction per second
        event[PREDICTIONS_PER_SECOND] = float(event[PREDICTIONS_COUNT_5M]) / 300
        base_fields = [TIMESTAMP, ENDPOINT_ID]

        base_event = {k: event[k] for k in base_fields}
        base_event[TIMESTAMP] = pd.to_datetime(
            base_event[TIMESTAMP], format=TIME_FORMAT
        )

        base_metrics = {
            RECORD_TYPE: BASE_METRICS,
            PREDICTIONS_PER_SECOND: event[PREDICTIONS_PER_SECOND],
            PREDICTIONS_COUNT_5M: event[PREDICTIONS_COUNT_5M],
            PREDICTIONS_COUNT_1H: event[PREDICTIONS_COUNT_1H],
            LATENCY_AVG_5M: event[LATENCY_AVG_5M],
            LATENCY_AVG_1H: event[LATENCY_AVG_1H],
            **base_event,
        }

        endpoint_features = {
            RECORD_TYPE: ENDPOINT_FEATURES,
            **event[NAMED_PREDICTIONS],
            **event[NAMED_FEATURES],
            **base_event,
        }

        processed = {BASE_METRICS: base_metrics, ENDPOINT_FEATURES: endpoint_features}

        if event[METRICS]:
            processed[CUSTOM_METRICS] = {
                RECORD_TYPE: CUSTOM_METRICS,
                **event[METRICS],
                **base_event,
            }

        return processed


class ProcessBeforeParquet(MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, event):
        logger.info("ProcessBeforeParquet1", event=event)
        for key in [UNPACKED_LABELS, FEATURES]:
            event.pop(key, None)
        value = event.get("entities")
        if value is not None:
            event = {**value, **event}
        for key in [LABELS, METRICS, ENTITIES]:
            if not event.get(key):
                event[key] = None
        logger.info("ProcessBeforeParquet2", event=event)
        return event


class ProcessEndpointEvent(MapClass):
    def __init__(self, kv_container: str, kv_path: str, v3io_access_key: str, **kwargs):
        super().__init__(**kwargs)
        self.kv_container: str = kv_container
        self.kv_path: str = kv_path
        self.v3io_access_key: str = v3io_access_key
        self.first_request: Dict[str, str] = dict()
        self.last_request: Dict[str, str] = dict()
        self.error_count: Dict[str, int] = defaultdict(int)
        self.endpoints: Set[str] = set()

    def do(self, full_event):
        event = full_event.body

        # code that calculates the endppint id. should be
        function_uri = event.get(FUNCTION_URI)
        if not is_not_none(function_uri, [FUNCTION_URI]):
            return None

        model = event.get(MODEL)
        if not is_not_none(model, [MODEL]):
            return None

        version = event.get(VERSION)
        versioned_model = f"{model}:{version}" if version else f"{model}:latest"

        endpoint_id = create_model_endpoint_id(
            function_uri=function_uri, versioned_model=versioned_model,
        )
        endpoint_id = str(endpoint_id)

        event[VERSIONED_MODEL] = versioned_model
        event[ENDPOINT_ID] = endpoint_id

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

        if not self.is_valid(endpoint_id, is_not_none, timestamp, ["when"],):
            return None

        if endpoint_id not in self.first_request:
            self.first_request[endpoint_id] = timestamp
        self.last_request[endpoint_id] = timestamp

        if not self.is_valid(endpoint_id, is_not_none, request_id, ["request", "id"],):
            return None
        if not self.is_valid(endpoint_id, is_not_none, latency, ["microsec"],):
            return None
        if not self.is_valid(
            endpoint_id, is_not_none, features, ["request", "inputs"],
        ):
            return None
        if not self.is_valid(
            endpoint_id, is_not_none, predictions, ["resp", "outputs"],
        ):
            return None

        unpacked_labels = {f"_{k}": v for k, v in event.get(LABELS, {}).items()}

        # Separate each model invocation into sub events
        events = []
        for i, (feature, prediction) in enumerate(zip(features, predictions)):
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
                    FUNCTION_URI: function_uri,
                    MODEL: versioned_model,
                    MODEL_CLASS: model_class,
                    TIMESTAMP: timestamp,
                    ENDPOINT_ID: endpoint_id,
                    REQUEST_ID: request_id,
                    LATENCY: latency,
                    FEATURES: feature,
                    PREDICTION: prediction,
                    FIRST_REQUEST: self.first_request[endpoint_id],
                    LAST_REQUEST: self.last_request[endpoint_id],
                    ERROR_COUNT: self.error_count[endpoint_id],
                    LABELS: event.get(LABELS, {}),
                    METRICS: event.get(METRICS, {}),
                    ENTITIES: event.get("request", {}).get(ENTITIES, {}),
                    UNPACKED_LABELS: unpacked_labels,
                }
            )

        storey_event = Event(body=events, key=endpoint_id, time=timestamp)
        return storey_event

    def is_list_of_numerics(
        self, field: List[Union[int, float, dict, list]], dict_path: List[str]
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
            if endpoint_record:
                first_request = endpoint_record.get(FIRST_REQUEST)
                if first_request:
                    self.first_request[endpoint_id] = first_request
                error_count = endpoint_record.get(ERROR_COUNT)
                if error_count:
                    self.error_count[endpoint_id] = error_count
            self.endpoints.add(endpoint_id)

    def is_valid(
        self, endpoint_id: str, validation_function, field: Any, dict_path: List[str]
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


def enrich_even_details(event) -> Optional[dict]:
    function_uri = event.get(FUNCTION_URI)

    if not is_not_none(function_uri, [FUNCTION_URI]):
        return None

    model = event.get(MODEL)
    if not is_not_none(model, [MODEL]):
        return None

    version = event.get(VERSION)
    versioned_model = f"{model}:{version}" if version else f"{model}:latest"

    endpoint_id = create_model_endpoint_id(
        function_uri=function_uri, versioned_model=versioned_model,
    )

    endpoint_id = str(endpoint_id)

    event[VERSIONED_MODEL] = versioned_model
    event[ENDPOINT_ID] = endpoint_id

    return event


def is_not_none(field: Any, dict_path: List[str]):
    if field is not None:
        return True
    logger.error(
        f"Expected event field is missing: {field} [Event -> {','.join(dict_path)}]"
    )
    return False


class FilterAndUnpackKeys(MapClass):
    def __init__(self, keys, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def do(self, event):
        new_event = {}
        for key in self.keys:
            if key in event:
                new_event[key] = event[key]
        unpacked = {}
        for key in new_event.keys():
            if key in self.keys:
                unpacked = {**unpacked, **new_event[key]}
            else:
                unpacked[key] = new_event[key]
        return unpacked if unpacked else None


class MapFeatureNames(MapClass):
    def __init__(
        self,
        kv_container: str,
        kv_path: str,
        access_key: str,
        infer_columns_from_data: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kv_container = kv_container
        self.kv_path = kv_path
        self.access_key = access_key
        self._infer_columns_from_data = infer_columns_from_data
        self.feature_names = {}
        self.label_columns = {}

    def _infer_feature_names_from_data(self, event):
        for endpoint_id in self.feature_names:
            if len(self.feature_names[endpoint_id]) >= len(event[FEATURES]):
                return self.feature_names[endpoint_id]
        return None

    def _infer_label_columns_from_data(self, event):
        for endpoint_id in self.label_columns:
            if len(self.label_columns[endpoint_id]) >= len(event[PREDICTION]):
                return self.label_columns[endpoint_id]
        return None

    def do(self, event: Dict):
        endpoint_id = event[ENDPOINT_ID]

        if endpoint_id not in self.feature_names:
            endpoint_record = get_endpoint_record(
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                endpoint_id=endpoint_id,
                access_key=self.access_key,
            )
            feature_names = endpoint_record.get(FEATURE_NAMES)
            feature_names = json.loads(feature_names) if feature_names else None

            label_columns = endpoint_record.get(LABEL_COLUMNS)
            label_columns = json.loads(label_columns) if label_columns else None

            if not feature_names and self._infer_columns_from_data:
                feature_names = self._infer_feature_names_from_data(event)

            if not feature_names:
                logger.warn(
                    "Feature names are not initialized, they will be automatically generated",
                    endpoint_id=endpoint_id,
                )
                feature_names = [f"f{i}" for i, _ in enumerate(event[FEATURES])]
                get_v3io_client().kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    access_key=self.access_key,
                    key=event[ENDPOINT_ID],
                    attributes={FEATURE_NAMES: json.dumps(feature_names)},
                    raise_for_status=RaiseForStatus.always,
                )

            if not label_columns and self._infer_columns_from_data:
                label_columns = self._infer_label_columns_from_data(event)

            if not label_columns:
                logger.warn(
                    "label column names are not initialized, they will be automatically generated",
                    endpoint_id=endpoint_id,
                )
                label_columns = [f"p{i}" for i, _ in enumerate(event[PREDICTION])]
                get_v3io_client().kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    access_key=self.access_key,
                    key=event[ENDPOINT_ID],
                    attributes={LABEL_COLUMNS: json.dumps(label_columns)},
                    raise_for_status=RaiseForStatus.always,
                )

            self.label_columns[endpoint_id] = label_columns
            self.feature_names[endpoint_id] = feature_names

            logger.info(
                "Label columns", endpoint_id=endpoint_id, label_columns=label_columns
            )
            logger.info(
                "Feature names", endpoint_id=endpoint_id, feature_names=feature_names
            )

        feature_names = self.feature_names[endpoint_id]
        features = event[FEATURES]
        event[NAMED_FEATURES] = {
            name: feature for name, feature in zip(feature_names, features)
        }

        label_columns = self.label_columns[endpoint_id]
        prediction = event[PREDICTION]
        event[NAMED_PREDICTIONS] = {
            name: prediction for name, prediction in zip(label_columns, prediction)
        }
        logger.info("Mapped event", event=event)
        return event


class WriteToKV(MapClass):
    def __init__(self, container: str, table: str, **kwargs):
        super().__init__(**kwargs)
        self.container = container
        self.table = table

    def do(self, event: Dict):
        get_v3io_client().kv.update(
            container=self.container,
            table_path=self.table,
            key=event[ENDPOINT_ID],
            attributes=event,
        )
        return event


class InferSchema(MapClass):
    def __init__(
        self,
        v3io_access_key: str,
        v3io_framesd: str,
        container: str,
        table: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.container = container
        self.v3io_access_key = v3io_access_key
        self.v3io_framesd = v3io_framesd
        self.table = table
        self.keys = set()

    def do(self, event: Dict):
        key_set = set(event.keys())
        if not key_set.issubset(self.keys):
            self.keys.update(key_set)
            get_frames_client(
                token=self.v3io_access_key,
                container=self.container,
                address=self.v3io_framesd,
            ).execute(backend="kv", table=self.table, command="infer_schema")
            logger.info(
                "Found new keys, inferred schema", table=self.table, event=event
            )
        return event


def get_endpoint_record(
    kv_container: str, kv_path: str, endpoint_id: str, access_key: str
) -> Optional[dict]:
    logger.info(
        "Grabbing endpoint data",
        container=kv_container,
        table_path=kv_path,
        key=endpoint_id,
    )
    try:
        endpoint_record = (
            get_v3io_client()
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
