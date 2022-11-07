# Copyright 2018 Iguazio
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

from fastapi import APIRouter, Depends

import mlrun.api.api.deps

from . import config, memory_reports

internal_router = APIRouter(
    prefix="/_internal",
    dependencies=[Depends(mlrun.api.api.deps.verify_api_state)],
    tags=["internal"],
)

internal_router.include_router(
    config.router,
    tags=["config"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)

internal_router.include_router(
    memory_reports.router,
    tags=["memory-reports"],
    dependencies=[
        Depends(mlrun.api.api.deps.authenticate_request),
        Depends(mlrun.api.api.deps.expose_internal_endpoints),
    ],
)
