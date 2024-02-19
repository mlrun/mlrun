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
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import os

def handler(context: nuclio.Context, event: nuclio.Event) -> None:
    """
    Run model monitoring application processor

    :param context: the MLRun context
    :param event:   trigger event
    """
    context.logger.info_with('[David] Got invoked',
                             trigger_kind=event.trigger.kind,
                             event_body=event.body,
                             )
    context.logger.info(f"[David] Event = {event.__repr__}")
    context.logger.info(f"[David] Context = {context.__dict__}")
    mlrun_context = mlrun.get_or_create_ctx("model_monitoring_controller")
    context.logger.info(f"[David] Mlrun Context = {mlrun_context.to_dict()}")
    if event.trigger.kind == 'cron':
        # log something
        context.logger.info('[David] Invoked from cron')
        context.logger.info(f'[David]  {event.trigger._struct["attributes"]["interval"]}')

    minutes = 1
    hours = days = 0
    batch_dict = {
        mm_constants.EventFieldType.MINUTES: minutes,
        mm_constants.EventFieldType.HOURS: hours,
        mm_constants.EventFieldType.DAYS: days,
    }
    mlrun_context.parameters[mm_constants.EventFieldType.BATCH_INTERVALS_DICT] = batch_dict
    monitor_app_controller = MonitoringApplicationController(
        context=mlrun_context,
        project=mlrun_context.project,
    )
    monitor_app_controller.run()
