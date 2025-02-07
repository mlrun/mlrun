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

from pydantic.v1 import validator
from pydantic.v1.dataclasses import dataclass

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.utils.v3io_clients
from mlrun.utils import logger

_RESULT_EXTRA_DATA_MAX_SIZE = 998


class _ModelMonitoringApplicationDataRes(ABC):
    name: str

    def __post_init__(self):
        pat = re.compile(mm_constants.RESULT_NAME_PATTERN)
        if not re.fullmatch(pat, self.name):
            raise mlrun.errors.MLRunValueError(
                f"Attribute name must comply with the regex `{mm_constants.RESULT_NAME_PATTERN}`"
            )

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError


@dataclass
class ModelMonitoringApplicationResult(_ModelMonitoringApplicationDataRes):
    """
    Class representing the result of a custom model monitoring application.

    :param name:           (str) Name of the application result. This name must be
                            unique for each metric in a single application
                            (name must be of the format :code:`[a-zA-Z_][a-zA-Z0-9_]*`).
    :param value:          (float) Value of the application result.
    :param kind:           (ResultKindApp) Kind of application result.
    :param status:         (ResultStatusApp) Status of the application result.
    :param extra_data:     (dict) Extra data associated with the application result. Note that if the extra data is
                                  exceeding the maximum size of 998 characters, it will be ignored and a message will
                                  be logged. In this case, we recommend logging the extra data as a separate artifact or
                                  shortening it.
    """

    name: str
    value: float
    kind: mm_constants.ResultKindApp
    status: mm_constants.ResultStatusApp
    extra_data: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mm_constants.ResultData.RESULT_NAME: self.name,
            mm_constants.ResultData.RESULT_VALUE: self.value,
            mm_constants.ResultData.RESULT_KIND: self.kind.value,
            mm_constants.ResultData.RESULT_STATUS: self.status.value,
            mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps(self.extra_data),
        }

    @validator("extra_data")
    @classmethod
    def validate_extra_data_len(cls, result_extra_data: dict):
        """Ensure that the extra data is not exceeding the maximum size which is important to avoid
        possible storage issues."""
        extra_data_len = len(json.dumps(result_extra_data))
        if extra_data_len > _RESULT_EXTRA_DATA_MAX_SIZE:
            logger.warning(
                f"Extra data is too long and won't be stored: {extra_data_len} characters while the maximum "
                f"is {_RESULT_EXTRA_DATA_MAX_SIZE} characters."
                f"Please shorten the extra data or log it as a separate artifact."
            )
            return {}
        return result_extra_data


@dataclass
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
            mm_constants.MetricData.METRIC_NAME: self.name,
            mm_constants.MetricData.METRIC_VALUE: self.value,
        }


@dataclasses.dataclass
class _ModelMonitoringApplicationStats(_ModelMonitoringApplicationDataRes):
    """
    Class representing the stats of histogram data drift application.

    :param name             (mm_constant.StatsKind) Enum mm_constant.StatsData of the stats data kind of the event
    :param                  (str) iso format representation of the timestamp the event took place
    :param stats            (dict) Dictionary representation of the stats calculated for the event

    """

    name: mm_constants.StatsKind
    timestamp: str
    stats: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mm_constants.StatsData.STATS_NAME: self.name,
            mm_constants.StatsData.STATS: self.stats,
            mm_constants.StatsData.TIMESTAMP: self.timestamp,
        }
