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
import mlrun.config
from mlrun.api.api.endpoints import (
    artifacts,
    auth,
    background_tasks,
    client_spec,
    clusterization_spec,
    feature_store,
    files,
    frontend_spec,
    functions,
    grafana_proxy,
    healthz,
    internal,
    logs,
    marketplace,
    model_endpoints,
    operations,
    pipelines,
    projects,
    runs,
    runtime_resources,
    schedules,
    secrets,
    submit,
    tags,
)

api_router = APIRouter(dependencies=[Depends(mlrun.api.api.deps.verify_api_state)])
api_router.include_router(
    artifacts.router,
    tags=["artifacts"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    auth.router,
    tags=["auth"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    background_tasks.router,
    tags=["background-tasks"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    files.router,
    tags=["files"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    functions.router,
    tags=["functions"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(healthz.router, tags=["healthz"])
api_router.include_router(client_spec.router, tags=["client-spec"])
api_router.include_router(clusterization_spec.router, tags=["clusterization-spec"])
api_router.include_router(
    logs.router,
    tags=["logs"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    pipelines.router,
    tags=["pipelines"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    projects.router,
    tags=["projects"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    runs.router,
    tags=["runs"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    runtime_resources.router,
    tags=["runtime-resources"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    schedules.router,
    tags=["schedules"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    submit.router,
    tags=["submit"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    feature_store.router,
    tags=["feature-store"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    frontend_spec.router,
    tags=["frontend-specs"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    secrets.router,
    tags=["secrets"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(grafana_proxy.router, tags=["grafana", "model-endpoints"])
api_router.include_router(model_endpoints.router, tags=["model-endpoints"])
api_router.include_router(
    marketplace.router,
    tags=["marketplace"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    operations.router,
    tags=["operations"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
api_router.include_router(
    tags.router,
    tags=["tags"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)

api_router.include_router(
    internal.internal_router,
    tags=["internal"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
