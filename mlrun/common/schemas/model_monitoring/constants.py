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

import hashlib
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

import mlrun.common.helpers
from mlrun.common.types import StrEnum


class EventFieldType:
    FUNCTION_URI = "function_uri"
    FUNCTION = "function"
    MODEL_URI = "model_uri"
    MODEL = "model"
    VERSION = "version"
    VERSIONED_MODEL = "versioned_model"
    MODEL_CLASS = "model_class"
    TIMESTAMP = "timestamp"
    # `endpoint_id` is deprecated as a field in the model endpoint schema since 1.3.1, replaced by `uid`.
    ENDPOINT_ID = "endpoint_id"
    UID = "uid"
    ENDPOINT_TYPE = "endpoint_type"
    REQUEST_ID = "request_id"
    RECORD_TYPE = "record_type"
    FEATURES = "features"
    FEATURE_NAMES = "feature_names"
    NAMED_FEATURES = "named_features"
    LABELS = "labels"
    LATENCY = "latency"
    LABEL_NAMES = "label_names"
    PREDICTION = "prediction"
    PREDICTIONS = "predictions"
    NAMED_PREDICTIONS = "named_predictions"
    ERROR_COUNT = "error_count"
    ENTITIES = "entities"
    FIRST_REQUEST = "first_request"
    LAST_REQUEST = "last_request"
    METRIC = "metric"
    METRICS = "metrics"
    BATCH_INTERVALS_DICT = "batch_intervals_dict"
    DEFAULT_BATCH_INTERVALS = "default_batch_intervals"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    MODEL_ENDPOINTS = "model_endpoints"
    STATE = "state"
    PROJECT = "project"
    STREAM_PATH = "stream_path"
    ACTIVE = "active"
    MONITORING_MODE = "monitoring_mode"
    FEATURE_STATS = "feature_stats"
    CURRENT_STATS = "current_stats"
    CHILDREN = "children"
    CHILDREN_UIDS = "children_uids"
    DRIFT_MEASURES = "drift_measures"
    DRIFT_STATUS = "drift_status"
    MONITOR_CONFIGURATION = "monitor_configuration"
    FEATURE_SET_URI = "monitoring_feature_set_uri"
    ALGORITHM = "algorithm"
    VALUE = "value"
    DRIFT_DETECTED_THRESHOLD = "drift_detected_threshold"
    POSSIBLE_DRIFT_THRESHOLD = "possible_drift_threshold"
    SAMPLE_PARQUET_PATH = "sample_parquet_path"


class ApplicationEvent:
    APPLICATION_NAME = "application_name"
    CURRENT_STATS = "current_stats"
    FEATURE_STATS = "feature_stats"
    SAMPLE_PARQUET_PATH = "sample_parquet_path"
    START_INFER_TIME = "start_infer_time"
    END_INFER_TIME = "end_infer_time"
    LAST_REQUEST = "last_request"
    ENDPOINT_ID = "endpoint_id"
    OUTPUT_STREAM_URI = "output_stream_uri"


class WriterEvent(StrEnum):
    APPLICATION_NAME = "application_name"
    ENDPOINT_ID = "endpoint_id"
    START_INFER_TIME = "start_infer_time"
    END_INFER_TIME = "end_infer_time"
    RESULT_NAME = "result_name"
    RESULT_VALUE = "result_value"
    RESULT_KIND = "result_kind"
    RESULT_STATUS = "result_status"
    RESULT_EXTRA_DATA = "result_extra_data"
    CURRENT_STATS = "current_stats"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class EventLiveStats:
    LATENCY_AVG_5M = "latency_avg_5m"
    LATENCY_AVG_1H = "latency_avg_1h"
    PREDICTIONS_PER_SECOND = "predictions_per_second"
    PREDICTIONS_COUNT_5M = "predictions_count_5m"
    PREDICTIONS_COUNT_1H = "predictions_count_1h"


class EventKeyMetrics:
    BASE_METRICS = "base_metrics"
    CUSTOM_METRICS = "custom_metrics"
    ENDPOINT_FEATURES = "endpoint_features"
    GENERIC = "generic"
    REAL_TIME = "real_time"


class TimeSeriesTarget:
    TSDB = "tsdb"


class ModelEndpointTarget:
    V3IO_NOSQL = "v3io-nosql"
    SQL = "sql"


class ProjectSecretKeys:
    ENDPOINT_STORE_CONNECTION = "MODEL_MONITORING_ENDPOINT_STORE_CONNECTION"
    ACCESS_KEY = "MODEL_MONITORING_ACCESS_KEY"
    PIPELINES_ACCESS_KEY = "MODEL_MONITORING_PIPELINES_ACCESS_KEY"
    KAFKA_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
    STREAM_PATH = "STREAM_PATH"


class ModelMonitoringStoreKinds:
    ENDPOINTS = "endpoints"
    EVENTS = "events"


class SchedulingKeys:
    LAST_ANALYZED = "last_analyzed"


class FileTargetKind:
    ENDPOINTS = "endpoints"
    EVENTS = "events"
    STREAM = "stream"
    PARQUET = "parquet"
    APPS_PARQUET = "apps_parquet"
    LOG_STREAM = "log_stream"


class ModelMonitoringMode(str, Enum):
    enabled = "enabled"
    disabled = "disabled"


class EndpointType(IntEnum):
    NODE_EP = 1  # end point that is not a child of a router
    ROUTER = 2  # endpoint that is router
    LEAF_EP = 3  # end point that is a child of a router


class PrometheusMetric:
    PREDICTIONS_TOTAL = "predictions_total"
    MODEL_LATENCY_SECONDS = "model_latency_seconds"
    INCOME_FEATURES = "income_features"
    ERRORS_TOTAL = "errors_total"
    DRIFT_METRICS = "drift_metrics"
    DRIFT_STATUS = "drift_status"


class MonitoringFunctionNames:
    WRITER = "model-monitoring-writer"
    BATCH = "model-monitoring-batch"
    APPLICATION_CONTROLLER = "model-monitoring-controller"
    STREAM = None

    @staticmethod
    def all():
        return [
            MonitoringFunctionNames.WRITER,
            MonitoringFunctionNames.STREAM,
            MonitoringFunctionNames.BATCH,
            MonitoringFunctionNames.APPLICATION_CONTROLLER,
        ]


@dataclass
class FunctionURI:
    project: str
    function: str
    tag: Optional[str] = None
    hash_key: Optional[str] = None

    @classmethod
    def from_string(cls, function_uri):
        project, uri, tag, hash_key = mlrun.common.helpers.parse_versioned_object_uri(
            function_uri
        )
        return cls(
            project=project,
            function=uri,
            tag=tag or None,
            hash_key=hash_key or None,
        )


@dataclass
class VersionedModel:
    model: str
    version: Optional[str]

    @classmethod
    def from_string(cls, model):
        try:
            model, version = model.split(":")
        except ValueError:
            model, version = model, None

        return cls(model, version)


@dataclass
class EndpointUID:
    project: str
    function: str
    function_tag: str
    function_hash_key: str
    model: str
    model_version: str
    uid: Optional[str] = None

    def __post_init__(self):
        function_ref = (
            f"{self.function}_{self.function_tag or self.function_hash_key or 'N/A'}"
        )
        versioned_model = f"{self.model}_{self.model_version or 'N/A'}"
        unique_string = f"{self.project}_{function_ref}_{versioned_model}"
        self.uid = hashlib.sha1(unique_string.encode("utf-8")).hexdigest()

    def __str__(self):
        return self.uid


class DriftStatus(Enum):
    """
    Enum for the drift status values.
    """

    NO_DRIFT = "NO_DRIFT"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    POSSIBLE_DRIFT = "POSSIBLE_DRIFT"


class ResultKindApp(Enum):
    """
    Enum for the result kind values
    """

    data_drift = 0
    concept_drift = 1
    model_performance = 2
    system_performance = 3


class ResultStatusApp(IntEnum):
    """
    Enum for the result status values, detected means that the app detected some problem.
    """

    irrelevant = -1
    no_detection = 0
    potential_detection = 1
    detected = 2


class ModelMonitoringAppLabel:
    KEY = "mlrun__type"
    VAL = "mlrun__model-monitoring-application"


class ControllerPolicy:
    BASE_PERIOD = "base_period"
