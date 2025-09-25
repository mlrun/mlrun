# Copyright 2025 Iguazio
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

import json

import mlrun.feature_store.steps
from mlrun.common.schemas.model_monitoring import (
    EventFieldType,
    EventKeyMetrics,
    StreamProcessingEvent,
    WriterEvent,
)
from mlrun.model_monitoring.db.tsdb.stream_graph_steps import BaseErrorExtractor


class ProcessBeforeTimescaleDB(mlrun.feature_store.steps.MapClass):
    """
    Process the data before writing to TimescaleDB. This step creates the relevant keys for the TimescaleDB table,
    including project name, custom metrics, time column, and table name column.

    :returns: Event as a dictionary
    """

    def do(self, event):
        event[EventFieldType.PROJECT] = event[EventFieldType.FUNCTION_URI].split("/")[0]
        event[EventKeyMetrics.CUSTOM_METRICS] = json.dumps(
            event.get(EventFieldType.METRICS, {})
        )
        # Map WHEN field to END_INFER_TIME for predictions data from model serving
        if StreamProcessingEvent.WHEN in event:
            event[WriterEvent.END_INFER_TIME] = event[StreamProcessingEvent.WHEN]
        # For non-prediction events, use timestamp as END_INFER_TIME to maintain consistency
        elif EventFieldType.TIMESTAMP in event:
            event[WriterEvent.END_INFER_TIME] = event[EventFieldType.TIMESTAMP]
        event[EventFieldType.TABLE_COLUMN] = f"_{event.get(EventFieldType.ENDPOINT_ID)}"

        return event


class TimescaleDBErrorExtractor(BaseErrorExtractor):
    """
    TimescaleDB-specific error extractor.

    Uses the shared base implementation with TimescaleDB-specific configuration.
    """

    pass
