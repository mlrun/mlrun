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
#

import logging
from enum import Enum

from pydantic import BaseModel, Field, validator

VALID_LOG_LEVELS = set(logging._nameToLevel.keys())
LogLevel = Enum("LogLevel", {level: level for level in VALID_LOG_LEVELS}, type=str)


class LogLevelMapping(BaseModel):
    domain_to_levels: dict[str, LogLevel] = Field(
        ...,
        example={"mlrun.api": "INFO"},
        description="Dictionary mapping log domains to log levels. Domains must start with 'mlrun'.",
    )
    recursive: bool = Field(
        False,
        description="Whether to set the log level for all sub-loggers of the specified domains.",
    )

    @validator("domain_to_levels")
    def validate_levels(cls, values: dict[str, LogLevel]) -> dict[str, LogLevel]:  # noqa: N805
        for domain, level in values.items():
            if not isinstance(domain, str):
                raise ValueError("Log domain must be a string")
            if not domain.startswith("mlrun"):
                raise ValueError(
                    f"Invalid log domain '{domain}'. "
                    f"Only domains starting with 'mlrun' are allowed."
                )
            if not isinstance(level, str):
                raise ValueError("Log level must be a string")
            if level.upper() not in VALID_LOG_LEVELS:
                raise ValueError(
                    f"Invalid log level '{level}' for domain '{domain}'. "
                    f"Allowed values are: {', '.join(sorted(VALID_LOG_LEVELS))}."
                )
        return values
