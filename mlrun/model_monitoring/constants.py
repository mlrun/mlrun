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
class EventFieldType:
    FUNCTION_URI = "function_uri"
    FUNCTION = "function"
    MODEL_URI = "model_uri"
    MODEL = "model"
    VERSION = "version"
    VERSIONED_MODEL = "versioned_model"
    MODEL_CLASS = "model_class"
    TIMESTAMP = "timestamp"
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
    LABEL_COLUMNS = "label_columns"
    LABEL_NAMES = "label_names"
    PREDICTION = "prediction"
    PREDICTIONS = "predictions"
    NAMED_PREDICTIONS = "named_predictions"
    ERROR_COUNT = "error_count"
    ENTITIES = "entities"
    FIRST_REQUEST = "first_request"
    LAST_REQUEST = "last_request"
    METRICS = "metrics"
    BATCH_TIMESTAMP = "batch_timestamp"
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    BATCH_INTERVALS_DICT = "batch_intervals_dict"
    DEFAULT_BATCH_INTERVALS = "default_batch_intervals"
    DEFAULT_BATCH_IMAGE = "default_batch_image"
    STREAM_IMAGE = "stream_image"
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
    ACCURACY = "accuracy"


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
    KV = "kv"
    SQL = "sql"
