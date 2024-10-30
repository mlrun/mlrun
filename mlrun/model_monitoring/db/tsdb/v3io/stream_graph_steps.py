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
from datetime import datetime
from typing import Any

import mlrun.feature_store.steps
from mlrun.common.schemas.model_monitoring import (
    EventFieldType,
    EventKeyMetrics,
    EventLiveStats,
)
from mlrun.utils import logger


def _normalize_dict_for_v3io_frames(event: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize user defined keys - input data to a model and its predictions,
    to a form V3IO frames tolerates.

    The dictionary keys should conform to '^[a-zA-Z_:]([a-zA-Z0-9_:])*$'.
    """
    prefix = "_"

    def norm_key(key: str) -> str:
        key = key.replace("-", "_")  # hyphens `-` are not allowed
        if key and key[0].isdigit():  # starting with a digit is not allowed
            return prefix + key
        return key

    return {norm_key(k): v for k, v in event.items()}


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
            EventFieldType.ENDPOINT_TYPE,
        ]

        # Getting event timestamp and endpoint_id
        base_event = {k: event[k] for k in base_fields}

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
            **_normalize_dict_for_v3io_frames(event[EventFieldType.NAMED_PREDICTIONS]),
            **_normalize_dict_for_v3io_frames(event[EventFieldType.NAMED_FEATURES]),
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


class ErrorExtractor(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Prepare the event for insertion into the errors TSDB table.
        """
        super().__init__(**kwargs)

    def do(self, event):
        error = event.get("error")
        timestamp = datetime.fromisoformat(event.get("when"))
        endpoint_id = event[EventFieldType.ENDPOINT_ID]
        event = {
            EventFieldType.MODEL_ERROR: str(error),
            EventFieldType.ERROR_TYPE: EventFieldType.INFER_ERROR,
            EventFieldType.ENDPOINT_ID: endpoint_id,
            EventFieldType.TIMESTAMP: timestamp,
            EventFieldType.ERROR_COUNT: 1.0,
        }
        logger.info("Write error to errors TSDB table", event=event)
        return event
