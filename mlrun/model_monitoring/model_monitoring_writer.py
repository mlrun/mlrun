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
#
import collections
import dataclasses
import datetime
import json
import os
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import v3io
import v3io.dataplane
import v3io_frames

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.data_types.infer
import mlrun.feature_store as fstore
import mlrun.utils.v3io_clients
from mlrun.utils import logger

from ..serving.utils import StepToDict

MODEL_MONITORING_WRITER_FUNCTION_NAME = "model-monitoring-writer"


class ModelMonitoringWriter(StepToDict):
    kind = "monitoring_application_stream_pusher"

    def __init__(self, name: str = None):
        self.name = name or "king"

    def do(self, event):

        print(
            f"endpoint_uid ={event['endpoint_uid']}, "
            f"app_name = {event['application_name']}, "
            f"schedule_time = {event['schedule_time']}, "
            f"result_name ={event['result_name']}, "
            f"result_value ={event['result_value']}, "
            f"my_name = {self.name}."
        )
        return event
