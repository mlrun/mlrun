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

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.feature_store.steps
from mlrun.utils import logger


class ProcessBeforeTDEngine(mlrun.feature_store.steps.MapClass):
    def __init__(self, **kwargs):
        """
        Process the data before writing to TDEngine. This step create the table name.

        :returns: Event as a dictionary which will be written into the TDEngine Metrics/Results tables.
        """
        super().__init__(**kwargs)

    def do(self, event):
        logger.info("Process event before writing to TDEngine", event=event)
        kind = event.get("kind")
        table_name = (
            f"{event[mm_schemas.WriterEvent.ENDPOINT_ID]}_"
            f"{event[mm_schemas.WriterEvent.APPLICATION_NAME]}"
        )
        if kind == mm_schemas.WriterEventKind.RESULT:
            # Write a new result
            event[mm_schemas.EventFieldType.TABLE_COLUMN] = (
                f"{table_name}_{event[mm_schemas.ResultData.RESULT_NAME]}"
            ).replace("-", "_")
        elif kind == mm_schemas.WriterEventKind.METRIC:
            # Write a new metric
            event[mm_schemas.EventFieldType.TABLE_COLUMN] = (
                f"{table_name}_{event[mm_schemas.MetricData.METRIC_NAME]}"
            ).replace("-", "_")
        event[mm_schemas.WriterEvent.START_INFER_TIME] = datetime.fromisoformat(
            event[mm_schemas.WriterEvent.START_INFER_TIME]
        )
        return event
