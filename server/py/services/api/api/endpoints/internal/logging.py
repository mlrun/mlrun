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

import fastapi

from mlrun.common.schemas.internal.logging import LogLevelMapping

router = fastapi.APIRouter()


@router.post("/log_levels")
async def set_log_levels(log_config: LogLevelMapping):
    for domain, level in log_config.domain_to_levels.items():
        numeric_level = getattr(logging, level.upper())
        logger_instance = logging.getLogger(domain)
        logger_instance.setLevel(numeric_level)
    return {"message": "Log levels updated successfully"}


@router.get(
    "/log_levels",
    response_model=LogLevelMapping,
)
async def get_log_levels():
    domain_to_levels = {}
    for name, logger_obj in logging.root.manager.loggerDict.items():
        if name.startswith("mlrun"):
            if isinstance(logger_obj, logging.Logger):
                level = logging.getLevelName(logger_obj.getEffectiveLevel())
                domain_to_levels[name] = level
    return {"domain_to_levels": domain_to_levels}
