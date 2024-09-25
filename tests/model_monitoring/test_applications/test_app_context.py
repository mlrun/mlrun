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

import inspect

import pytest

from mlrun.model_monitoring.applications.context import MonitoringApplicationContext
from mlrun.projects import MlrunProject


@pytest.mark.parametrize("method", ["log_artifact", "log_dataset", "log_model"])
def test_log_object_signature(method: str) -> None:
    """Future-proof the `log_x` method of MM app context with respect to the project object"""
    assert inspect.signature(
        getattr(MonitoringApplicationContext, method)
    ) == inspect.signature(getattr(MlrunProject, method))
