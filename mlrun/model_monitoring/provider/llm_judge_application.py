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

#This is a wrapper class for the mlrun to interact with the huggingface's evaluate library
import uuid
from typing import Union, List
from mlrun.model_monitoring.application import ModelMonitoringApplication, ModelMonitoringApplicationResult

_HAS_evaluate = False
try:
    import evaluate # noqa: F401
    _HAS_evaluate = True
except ModuleNotFoundError:
    pass

if _HAS_evaluate:
    import evaluate

class HFEvaluateApplication(ModelMonitoringApplication):
