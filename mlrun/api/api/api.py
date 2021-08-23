from fastapi import APIRouter, Depends

from mlrun.api.api import deps
from mlrun.api.api.endpoints import (
    artifacts,
    auth,
    background_tasks,
    client_spec,
    feature_store,
    files,
    frontend_spec,
    functions,
    grafana_proxy,
    healthz,
    logs,
    marketplace,
    model_endpoints,
    pipelines,
    projects,
    runs,
    runtime_resources,
    schedules,
    secrets,
    submit,
)

api_router = APIRouter()
api_router.include_router(
    artifacts.router, tags=["artifacts"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    auth.router, tags=["auth"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    background_tasks.router,
    tags=["background-tasks"],
    dependencies=[Depends(deps.AuthVerifierDep)],
)
api_router.include_router(
    files.router, tags=["files"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    functions.router, tags=["functions"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(healthz.router, tags=["healthz"])
api_router.include_router(client_spec.router, tags=["client-spec"])
api_router.include_router(
    logs.router, tags=["logs"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    pipelines.router, tags=["pipelines"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    projects.router, tags=["projects"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    runs.router, tags=["runs"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    runtime_resources.router,
    tags=["runtime-resources"],
    dependencies=[Depends(deps.AuthVerifierDep)],
)
api_router.include_router(
    schedules.router, tags=["schedules"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    submit.router, tags=["submit"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(
    feature_store.router,
    tags=["feature-store"],
    dependencies=[Depends(deps.AuthVerifierDep)],
)
api_router.include_router(
    frontend_spec.router,
    tags=["frontend-specs"],
    dependencies=[Depends(deps.AuthVerifierDep)],
)
api_router.include_router(
    secrets.router, tags=["secrets"], dependencies=[Depends(deps.AuthVerifierDep)]
)
api_router.include_router(grafana_proxy.router, tags=["grafana", "model-endpoints"])
api_router.include_router(model_endpoints.router, tags=["model-endpoints"])
api_router.include_router(
    marketplace.router,
    tags=["marketplace"],
    dependencies=[Depends(deps.AuthVerifierDep)],
)
