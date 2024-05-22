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
#

import json

import mlrun.feature_store.steps
from mlrun.common.schemas.model_monitoring import (
    EventFieldType,
    EventKeyMetrics,
)

_TABLE_COLUMN = "table_column"


class ProcessBeforeTDEngine(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Process the data before writing to TDEngine. This step create the relevant keys for the TDEngine table,
        including project name, custom metrics, time column, and table name column.

        :returns: Event as a dictionary which will be written into the TDEngine Predictions table.
        """
        super().__init__(**kwargs)

    def do(self, event):
        event[EventFieldType.PROJECT] = event[EventFieldType.FUNCTION_URI].split("/")[0]
        event[EventKeyMetrics.CUSTOM_METRICS] = json.dumps(
            event.get(EventFieldType.METRICS, {})
        )
        event[EventFieldType.TIME] = event.get(EventFieldType.TIMESTAMP)
        event[EventFieldType.TABLE_COLUMN] = "_" + event.get(EventFieldType.ENDPOINT_ID)

        return event
