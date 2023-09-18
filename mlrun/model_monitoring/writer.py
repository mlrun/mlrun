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
import mlrun.common.schemas
from mlrun.serving.utils import StepToDict


class ModelMonitoringWriter(StepToDict):
    """DEMO WRITER TODO"""

    kind = "monitoring_application_stream_pusher"

    def __init__(self, name: str = None):
        self.name = name or "king"

    def do(self, event):

        print(
            f"endpoint_uid ={event[mlrun.common.schemas.model_monitoring.constants.WriterEvent.ENDPOINT_ID]}, \n"
            f"app_name = {event[mlrun.common.schemas.model_monitoring.constants.WriterEvent.APPLICATION_NAME]}, \n"
            f"schedule_time = {event[mlrun.common.schemas.model_monitoring.constants.WriterEvent.SCHEDULE_TIME]}, \n"
            f"result_name ={event[mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_NAME]}, \n"
            f"result_value ={event[mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_VALUE]}, \n"
            f"my_name = {self.name}."
        )
        return event
