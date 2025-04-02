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
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional

import mlrun.common.constants
import mlrun.common.helpers
from mlrun.common.types import StrEnum


class MonitoringStrEnum(StrEnum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ModelEndpointSchema(MonitoringStrEnum):
    # metadata
    UID = "uid"
    PROJECT = "project"
    ENDPOINT_TYPE = "endpoint_type"
    NAME = "name"
    CREATED = "created"
    UPDATED = "updated"
    LABELS = "labels"

    # spec
    FUNCTION_NAME = "function_name"
    FUNCTION_TAG = "function_tag"
    MODEL_NAME = "model_name"
    MODEL_TAGS = "model_tags"
    MODEL_PATH = "model_path"
    MODEL_CLASS = "model_class"
    FEATURE_NAMES = "feature_names"
    LABEL_NAMES = "label_names"
    FEATURE_STATS = "feature_stats"
    MONITORING_FEATURE_SET_URI = "monitoring_feature_set_uri"
    CHILDREN = "children"
    CHILDREN_UIDS = "children_uids"
    FUNCTION_URI = "function_uri"
    MODEL_URI = "model_uri"

    # status
    STATE = "state"
    MONITORING_MODE = "monitoring_mode"
    FIRST_REQUEST = "first_request"
    SAMPLING_PERCENTAGE = "sampling_percentage"

    # status - operative
    LAST_REQUEST = "last_request"
    RESULT_STATUS = "result_status"
    AVG_LATENCY = "avg_latency"
    ERROR_COUNT = "error_count"
    CURRENT_STATS = "current_stats"
    DRIFT_MEASURES = "drift_measures"


class ModelEndpointCreationStrategy(MonitoringStrEnum):
    INPLACE = "inplace"
    ARCHIVE = "archive"
    OVERWRITE = "overwrite"
    SKIP = "skip"


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
    ENDPOINT_NAME = "endpoint_name"
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
    MODEL_ERROR = "model_error"
    ERROR_TYPE = "error_type"
    INFER_ERROR = "infer_error"
    ENTITIES = "entities"
    FIRST_REQUEST = "first_request"
    LAST_REQUEST = "last_request"
    LAST_REQUEST_TIMESTAMP = "last_request_timestamp"
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
    SAMPLE_PARQUET_PATH = "sample_parquet_path"
    TIME = "time"
    TABLE_COLUMN = "table_column"
    SAMPLING_PERCENTAGE = "sampling_percentage"
    SAMPLING_RATE = "sampling_rate"
    ESTIMATED_PREDICTION_COUNT = "estimated_prediction_count"
    EFFECTIVE_SAMPLE_COUNT = "effective_sample_count"


class FeatureSetFeatures(MonitoringStrEnum):
    LATENCY = EventFieldType.LATENCY
    METRICS = EventFieldType.METRICS

    @classmethod
    def time_stamp(cls):
        return EventFieldType.TIMESTAMP

    @classmethod
    def entity(cls):
        return EventFieldType.ENDPOINT_ID


class ApplicationEvent:
    APPLICATION_NAME = "application_name"
    START_INFER_TIME = "start_infer_time"
    END_INFER_TIME = "end_infer_time"
    ENDPOINT_ID = "endpoint_id"
    ENDPOINT_NAME = "endpoint_name"
    ENDPOINT_UPDATED = "endpoint_updated"


class WriterEvent(MonitoringStrEnum):
    ENDPOINT_NAME = "endpoint_name"
    APPLICATION_NAME = "application_name"
    ENDPOINT_ID = "endpoint_id"
    START_INFER_TIME = "start_infer_time"
    END_INFER_TIME = "end_infer_time"
    EVENT_KIND = "event_kind"  # metric or result or stats
    DATA = "data"


class WriterEventKind(MonitoringStrEnum):
    METRIC = "metric"
    RESULT = "result"
    STATS = "stats"


class ControllerEvent(MonitoringStrEnum):
    KIND = "kind"
    ENDPOINT_ID = "endpoint_id"
    ENDPOINT_NAME = "endpoint_name"
    PROJECT = "project"
    TIMESTAMP = "timestamp"
    FIRST_REQUEST = "first_request"
    FEATURE_SET_URI = "feature_set_uri"
    ENDPOINT_TYPE = "endpoint_type"
    ENDPOINT_POLICY = "endpoint_policy"
    # Note: currently under endpoint policy we will have a dictionary including the keys: "application_names"
    # "base_period", and "updated_endpoint" stand for when the MEP was updated


class ControllerEventEndpointPolicy(MonitoringStrEnum):
    BASE_PERIOD = "base_period"
    MONITORING_APPLICATIONS = "monitoring_applications"
    ENDPOINT_UPDATED = "endpoint_updated"


class ControllerEventKind(MonitoringStrEnum):
    NOP_EVENT = "nop_event"
    REGULAR_EVENT = "regular_event"


class MetricData(MonitoringStrEnum):
    METRIC_NAME = "metric_name"
    METRIC_VALUE = "metric_value"


class ResultData(MonitoringStrEnum):
    RESULT_NAME = "result_name"
    RESULT_VALUE = "result_value"
    RESULT_KIND = "result_kind"
    RESULT_STATUS = "result_status"
    RESULT_EXTRA_DATA = "result_extra_data"


class StatsData(MonitoringStrEnum):
    STATS_NAME = "stats_name"
    STATS = "stats"
    TIMESTAMP = "timestamp"


class StatsKind(MonitoringStrEnum):
    CURRENT_STATS = "current_stats"
    DRIFT_MEASURES = "drift_measures"


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


class TSDBTarget(MonitoringStrEnum):
    V3IO_TSDB = "v3io-tsdb"
    TDEngine = "tdengine"


class ProjectSecretKeys:
    ACCESS_KEY = "MODEL_MONITORING_ACCESS_KEY"
    TSDB_PROFILE_NAME = "TSDB_PROFILE_NAME"
    STREAM_PROFILE_NAME = "STREAM_PROFILE_NAME"

    @classmethod
    def mandatory_secrets(cls):
        return [
            cls.STREAM_PROFILE_NAME,
            cls.TSDB_PROFILE_NAME,
        ]


class GetEventsFormat(MonitoringStrEnum):
    SINGLE = "single"
    SEPARATION = "separation"
    INTERSECTION = "intersection"


class FileTargetKind:
    ENDPOINTS = "endpoints"
    EVENTS = "events"
    PREDICTIONS = "predictions"
    STREAM = "stream"
    PARQUET = "parquet"
    APPS_PARQUET = "apps_parquet"
    LOG_STREAM = "log_stream"
    MONITORING_SCHEDULES = "monitoring_schedules"
    MONITORING_APPLICATION = "monitoring_application"
    ERRORS = "errors"
    STATS = "stats"
    LAST_REQUEST = "last_request"


class ModelMonitoringMode(StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class ScheduleChiefFields(StrEnum):
    LAST_REQUEST = "last_request"
    LAST_ANALYZED = "last_analyzed"


class EndpointType(IntEnum):
    NODE_EP = 1  # end point that is not a child of a router
    ROUTER = 2  # endpoint that is router
    LEAF_EP = 3  # end point that is a child of a router
    BATCH_EP = 4  # endpoint that is representing an offline batch endpoint

    @classmethod
    def top_level_list(cls):
        return [cls.NODE_EP, cls.ROUTER, cls.BATCH_EP]


class MonitoringFunctionNames(MonitoringStrEnum):
    STREAM = "model-monitoring-stream"
    APPLICATION_CONTROLLER = "model-monitoring-controller"
    WRITER = "model-monitoring-writer"


class V3IOTSDBTables(MonitoringStrEnum):
    APP_RESULTS = "app-results"
    METRICS = "metrics"
    EVENTS = "events"
    ERRORS = "errors"
    PREDICTIONS = "predictions"


class TDEngineSuperTables(MonitoringStrEnum):
    APP_RESULTS = "app_results"
    METRICS = "metrics"
    PREDICTIONS = "predictions"
    ERRORS = "errors"


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
    uid: str = field(init=False)

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
    mm_app_anomaly = 4


class ResultStatusApp(IntEnum):
    """
    Enum for the result status values, detected means that the app detected some problem.
    """

    irrelevant = -1
    no_detection = 0
    potential_detection = 1
    detected = 2


class ModelMonitoringAppLabel:
    KEY = mlrun.common.constants.MLRunInternalLabels.mlrun_type
    VAL = "mlrun__model-monitoring-application"

    def __str__(self) -> str:
        return f"{self.KEY}={self.VAL}"


class HistogramDataDriftApplicationConstants:
    NAME = "histogram-data-drift"
    GENERAL_RESULT_NAME = "general_drift"


class PredictionsQueryConstants:
    DEFAULT_AGGREGATION_GRANULARITY = "10m"
    INVOCATIONS = "invocations"


class SpecialApps:
    MLRUN_INFRA = "mlrun-infra"


_RESERVED_FUNCTION_NAMES = MonitoringFunctionNames.list() + [SpecialApps.MLRUN_INFRA]


class ModelEndpointMonitoringMetricType(StrEnum):
    RESULT = "result"
    METRIC = "metric"


_FQN_PART_PATTERN = r"[a-zA-Z0-9_-]+"
FQN_PATTERN = (
    rf"^(?P<project>{_FQN_PART_PATTERN})\."
    rf"(?P<app>{_FQN_PART_PATTERN})\."
    rf"(?P<type>{ModelEndpointMonitoringMetricType.RESULT}|{ModelEndpointMonitoringMetricType.METRIC})\."
    rf"(?P<name>{_FQN_PART_PATTERN})$"
)
FQN_REGEX = re.compile(FQN_PATTERN)

# refer to `mlrun.utils.regex.project_name`
PROJECT_PATTERN = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"
MODEL_ENDPOINT_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
RESULT_NAME_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_]*"

INTERSECT_DICT_KEYS = {
    ModelEndpointMonitoringMetricType.METRIC: "intersect_metrics",
    ModelEndpointMonitoringMetricType.RESULT: "intersect_results",
}

CRON_TRIGGER_KINDS = ("http", "cron")
STREAM_TRIGGER_KINDS = ("v3io-stream", "kafka-cluster")
