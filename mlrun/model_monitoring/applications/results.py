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

import dataclasses
import json
import re
from abc import ABC, abstractmethod

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.utils.v3io_clients


class _ModelMonitoringApplicationDataRes(ABC):
    name: str

    def __post_init__(self):
        pat = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
        if not re.fullmatch(pat, self.name):
            raise mlrun.errors.MLRunValueError(
                "Attribute name must comply with the regex `[a-zA-Z_][a-zA-Z0-9_]*`"
            )

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError


@dataclasses.dataclass
class ModelMonitoringApplicationResult(_ModelMonitoringApplicationDataRes):
    """
    Class representing the result of a custom model monitoring application.

    :param name:           (str) Name of the application result. This name must be
                            unique for each metric in a single application
                            (name must be of the format :code:`[a-zA-Z_][a-zA-Z0-9_]*`).
    :param value:          (float) Value of the application result.
    :param kind:           (ResultKindApp) Kind of application result.
    :param status:         (ResultStatusApp) Status of the application result.
    :param extra_data:     (dict) Extra data associated with the application result.
    """

    name: str
    value: float
    kind: mm_constant.ResultKindApp
    status: mm_constant.ResultStatusApp
    extra_data: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mm_constant.ResultData.RESULT_NAME: self.name,
            mm_constant.ResultData.RESULT_VALUE: self.value,
            mm_constant.ResultData.RESULT_KIND: self.kind.value,
            mm_constant.ResultData.RESULT_STATUS: self.status.value,
            mm_constant.ResultData.RESULT_EXTRA_DATA: json.dumps(self.extra_data),
        }


@dataclasses.dataclass
class ModelMonitoringApplicationMetric(_ModelMonitoringApplicationDataRes):
    """
    Class representing a single metric of a custom model monitoring application.

    :param name:           (str) Name of the application metric. This name must be
                            unique for each metric in a single application
                            (name must be of the format :code:`[a-zA-Z_][a-zA-Z0-9_]*`).
    :param value:          (float) Value of the application metric.
    """

    name: str
    value: float

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mm_constant.MetricData.METRIC_NAME: self.name,
            mm_constant.MetricData.METRIC_VALUE: self.value,
        }
