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

import collections
import datetime
import json
import os
import re
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import requests
import v3io
import v3io.dataplane
import v3io_frames
from v3io_frames.frames_pb2 import IGNORE

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.data_types.infer
import mlrun.feature_store as fstore
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.helpers import calculate_inputs_statistics
from mlrun.model_monitoring.metrics.histogram_distance import (
    HellingerDistance,
    HistogramDistanceMetric,
    KullbackLeiblerDivergence,
    TotalVarianceDistance,
)
from mlrun.utils import logger

# A type for representing a drift result, a tuple of the status and the drift mean:
DriftResultType = tuple[mlrun.common.schemas.model_monitoring.DriftStatus, float]
