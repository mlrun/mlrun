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

from datetime import datetime

import mlrun.feature_store.steps
from mlrun.common.schemas.model_monitoring import EventFieldType

# Import the authoritative database schema constant
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema import (
    MODEL_ERROR_MAX_LENGTH,
)
from mlrun.utils import logger

# Error truncation log message
ERROR_TRUNCATION_MESSAGE = "Error message truncated for storage"


class BaseErrorExtractor(mlrun.feature_store.steps.MapClass):
    """
    Shared error extraction implementation for TDEngine and TimescaleDB.

    Prepares events for insertion into the errors TSDB table.
    These two implementations are identical and can use this shared base class.
    V3io has different requirements and uses its own implementation.
    """

    def do(self, event):
        error = str(event.get("error"))
        original_error_length = len(error)
        if len(error) > MODEL_ERROR_MAX_LENGTH:
            error = error[-MODEL_ERROR_MAX_LENGTH:]
            logger.warning(
                ERROR_TRUNCATION_MESSAGE,
                endpoint_id=event.get(EventFieldType.ENDPOINT_ID),
                function_uri=event.get(EventFieldType.FUNCTION_URI),
                original_error_length=original_error_length,
                max_length=MODEL_ERROR_MAX_LENGTH,
                truncated_error=error,
            )
        timestamp = datetime.fromisoformat(event.get("when"))
        endpoint_id = event[EventFieldType.ENDPOINT_ID]
        result_event = {
            EventFieldType.MODEL_ERROR: error,
            EventFieldType.ERROR_TYPE: EventFieldType.INFER_ERROR,
            EventFieldType.ENDPOINT_ID: endpoint_id,
            EventFieldType.TIME: timestamp,
            EventFieldType.PROJECT: event[EventFieldType.FUNCTION_URI].split("/")[0],
            EventFieldType.TABLE_COLUMN: "_err_"
            + event.get(EventFieldType.ENDPOINT_ID),
        }
        logger.info("Write error to errors TSDB table", event=result_event)
        return result_event
