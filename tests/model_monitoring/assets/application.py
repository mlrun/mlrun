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

import mlrun.model_monitoring.applications as mm_applications


class DemoMonitoringApp(mm_applications.ModelMonitoringApplicationBase):
    _dict_fields = ["param_1", "param_2"]

    def __init__(self, param_1, **kwargs) -> None:
        self.param_1 = param_1
        self.param_2 = kwargs["param_2"]

    def do_tracking(
        self,
        monitoring_context,
    ):
        pass
