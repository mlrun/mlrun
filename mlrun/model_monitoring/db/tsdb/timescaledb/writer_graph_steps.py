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
from datetime import datetime

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.feature_store.steps
from mlrun.utils import logger


class ProcessBeforeTimescaleDBWriter(mlrun.feature_store.steps.MapClass):
    """
    Process the data before writing to TimescaleDB via the new async writer.

    This step combines functionality from both the existing stream processor
    and the TDEngine writer pattern to create appropriate table names and
    format data for TimescaleDB writer targets.

    :returns: Event as a dictionary which will be written into the TimescaleDB Metrics/App Results tables.
    """

    def do(self, event):
        logger.info("Process event before writing to TimescaleDB writer", event=event)

        # Extract project from function URI (existing TimescaleDB pattern)
        if mm_schemas.EventFieldType.FUNCTION_URI in event:
            event[mm_schemas.EventFieldType.PROJECT] = event[
                mm_schemas.EventFieldType.FUNCTION_URI
            ].split("/")[0]

        # Handle custom metrics serialization (existing TimescaleDB pattern)
        event[mm_schemas.EventKeyMetrics.CUSTOM_METRICS] = json.dumps(
            event.get(mm_schemas.EventFieldType.METRICS, {})
        )

        # Handle time mapping (existing TimescaleDB pattern)
        # Map WHEN field to END_INFER_TIME for predictions data from model serving
        if mm_schemas.StreamProcessingEvent.WHEN in event:
            event[mm_schemas.WriterEvent.END_INFER_TIME] = event[
                mm_schemas.StreamProcessingEvent.WHEN
            ]
        # For non-prediction events, use timestamp as END_INFER_TIME to maintain consistency
        elif mm_schemas.EventFieldType.TIMESTAMP in event:
            event[mm_schemas.WriterEvent.END_INFER_TIME] = event[
                mm_schemas.EventFieldType.TIMESTAMP
            ]

        # Handle START_INFER_TIME conversion (TDEngine pattern)
        if mm_schemas.WriterEvent.START_INFER_TIME in event and isinstance(
            event[mm_schemas.WriterEvent.START_INFER_TIME], str
        ):
            event[mm_schemas.WriterEvent.START_INFER_TIME] = datetime.fromisoformat(
                event[mm_schemas.WriterEvent.START_INFER_TIME]
            )

        # Create table column identifier (adapted from both patterns)
        # TimescaleDB uses endpoint-based table organization unlike TDEngine's complex naming
        event[mm_schemas.EventFieldType.TABLE_COLUMN] = (
            f"_{event.get(mm_schemas.EventFieldType.ENDPOINT_ID)}"
        )

        return event
