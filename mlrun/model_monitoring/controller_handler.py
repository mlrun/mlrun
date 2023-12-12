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

import mlrun
from mlrun.model_monitoring.controller import MonitoringApplicationController


def handler(context: mlrun.run.MLClientCtx):
    """
    Run model monitoring application processor

    :param context: the MLRun context
    """
    monitor_app_controller = MonitoringApplicationController(
        context=context,
        project=context.project,
    )
    monitor_app_controller.run()
    if monitor_app_controller.endpoints_exceptions:
        context.logger.error(monitor_app_controller.endpoints_exceptions)
