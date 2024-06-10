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
from fastapi import APIRouter, Depends

from server.api.api import deps
from server.api.api.endpoints import (
    alert_template,
    alerts,
    artifacts,
    artifacts_v2,
    auth,
    background_tasks,
    client_spec,
    clusterization_spec,
    datastore_profile,
    events,
    feature_store,
    feature_store_v2,
    files,
    frontend_spec,
    functions,
    functions_v2,
    grafana_proxy,
    healthz,
    hub,
    internal,
    jobs,
    logs,
    model_endpoints,
    model_monitoring,
    nuclio,
    operations,
    pipelines,
    projects,
    projects_v2,
    runs,
    runtime_resources,
    schedules,
    secrets,
    submit,
    tags,
    workflows,
)

# v1 router
api_router = APIRouter(dependencies=[Depends(deps.verify_api_state)])
api_router.include_router(
    artifacts.router,
    tags=["artifacts"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    auth.router,
    tags=["auth"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    background_tasks.router,
    tags=["background-tasks"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    files.router,
    tags=["files"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    functions.router,
    tags=["functions"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(healthz.router, tags=["healthz"])
api_router.include_router(client_spec.router, tags=["client-spec"])
api_router.include_router(clusterization_spec.router, tags=["clusterization-spec"])
api_router.include_router(
    logs.router,
    tags=["logs"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    pipelines.router,
    tags=["pipelines"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    projects.router,
    tags=["projects"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    runs.router,
    tags=["runs"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    runtime_resources.router,
    tags=["runtime-resources"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    schedules.router,
    tags=["schedules"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    submit.router,
    tags=["submit"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    feature_store.router,
    tags=["feature-store"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    feature_store_v2.router,
    tags=["feature-store"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    frontend_spec.router,
    tags=["frontend-specs"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    secrets.router,
    tags=["secrets"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(grafana_proxy.router, tags=["grafana", "model-endpoints"])
api_router.include_router(model_endpoints.router, tags=["model-endpoints"])
api_router.include_router(model_monitoring.router, tags=["model-monitoring"])
api_router.include_router(jobs.router, tags=["jobs"])
api_router.include_router(
    hub.router,
    tags=["hub"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    operations.router,
    tags=["operations"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    tags.router,
    tags=["tags"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    internal.internal_router,
    tags=["internal"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    events.router,
    tags=["events"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    alerts.router,
    tags=["alerts"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    alert_template.router,
    tags=["alert-templates"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    workflows.router,
    tags=["workflows"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    datastore_profile.router,
    tags=["datastores"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_router.include_router(
    nuclio.router,
    tags=["nuclio"],
    dependencies=[Depends(deps.authenticate_request)],
)

# v2 Router
api_v2_router = APIRouter(dependencies=[Depends(deps.verify_api_state)])
api_v2_router.include_router(
    healthz.router,
    tags=["healthz"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_v2_router.include_router(
    artifacts_v2.router,
    tags=["artifacts"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_v2_router.include_router(
    projects_v2.router,
    tags=["projects"],
    dependencies=[Depends(deps.authenticate_request)],
)
api_v2_router.include_router(
    functions_v2.router,
    tags=["functions"],
    dependencies=[Depends(deps.authenticate_request)],
)
