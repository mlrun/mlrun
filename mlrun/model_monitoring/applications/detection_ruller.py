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

from typing import Optional

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.utils.v3io_clients
from mlrun.datastore import get_stream_pusher
from mlrun.model_monitoring.helpers import get_stream_path
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

from ..application import ModelMonitoringApplicationResult
from .context import MonitoringApplicationContext
from mlrun.model import EntrypointParam, ImageBuilder, ModelObj


class Ruler(ModelObj):
    pass
    # name: str
    # status -> name, val (?), color, description
    # logic query
    # numeric query (to present also float value/ )
    # names of relevant metrics (?)
    # action to take(alerts) list of status names -> bring up an alert (?)


class RulerOverTime(Ruler):
    pass
    # name
    # time range
    # Gradual, Sudden, Blip, Recurrent, No Drift
    #
    # Alerting
