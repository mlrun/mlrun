# Copyright 2025 Iguazio
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

import enum
from datetime import datetime
from typing import Optional

from pydantic.v1 import BaseModel


class FunctionsType(enum.Enum):
    APPLICATION = "application"
    INFRA = "infra"


class FunctionSummary(BaseModel):
    """
    Function summary model. Includes metadata about the function, such as its name, as well as statistical
    metrics such as the number of detections and possible detections. A function summary can be from either a
    model monitoring application (type "application") or an infrastructure function (type "infra").
    """

    type: FunctionsType
    name: str
    application_class: str
    project_name: str
    updated_time: datetime
    status: Optional[str] = None
    base_period: Optional[int] = None
    stats: Optional[dict] = None

    @classmethod
    def from_function_dict(
        cls,
        func_dict: dict,
        func_type=FunctionsType.APPLICATION,
        base_period: Optional[int] = None,
        stats: Optional[dict] = None,
    ):
        """
        Create a FunctionSummary instance from a dictionary.
        """

        return cls(
            type=func_type,
            name=func_dict["metadata"]["name"],
            application_class=""
            if func_type != FunctionsType.APPLICATION
            else func_dict["spec"]["graph"]["steps"]["PushToMonitoringWriter"]["after"][
                0
            ],
            project_name=func_dict["metadata"]["project"],
            updated_time=func_dict["metadata"].get("updated"),
            status=func_dict["status"].get("state"),
            base_period=base_period,
            stats=stats,
        )
