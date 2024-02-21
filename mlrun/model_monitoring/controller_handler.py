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
import nuclio

import mlrun
from mlrun.model_monitoring.controller import MonitoringApplicationController


def handler(context: nuclio.Context, event: nuclio.Event) -> None:
    """
    Run model monitoring application processor

    :param context: the Nuclio context
    :param event:   trigger event
    """
    context.user_data.monitor_app_controller.run(event)


def init_context(context):
    mlrun_context = mlrun.get_or_create_ctx("model_monitoring_controller")
    mlrun_context.logger.info("Initialize monitoring app controller")
    monitor_app_controller = MonitoringApplicationController(
        mlrun_context=mlrun_context,
        project=mlrun_context.project,
    )
    setattr(context.user_data, "monitor_app_controller", monitor_app_controller)
