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

from pydantic import BaseModel, root_validator

VALID_LOG_LEVELS = set(logging._nameToLevel.keys())


class LogLevelMapping(BaseModel):
    __root__: dict[str, str]

    @root_validator(pre=True)
    def validate_levels(cls, values):  # noqa: N805
        for domain, level in values.items():
            if not isinstance(domain, str):
                raise ValueError("Log domain must be a string")
            if not isinstance(level, str):
                raise ValueError("Log level must be a string")
            if level.upper() not in VALID_LOG_LEVELS:
                raise ValueError(
                    f"Invalid log level '{level}' for domain '{domain}'. "
                    f"Allowed values are: {', '.join(sorted(VALID_LOG_LEVELS))}."
                )
        return values

    @property
    def log_items(self) -> dict[str, str]:
        """Getter to return the log items (domain-level mappings)."""
        return self.__root__
