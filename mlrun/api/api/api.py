from fastapi import APIRouter, Depends

from mlrun.api.api import deps
from mlrun.api.api.endpoints import (
    artifacts,
    background_tasks,
    feature_store,
    files,
    frontend_spec,
    functions,
    grafana_proxy,
    healthz,
    logs,
    model_endpoints,
    pipelines,
    projects,
    runs,
    runtimes,
    schedules,
    secrets,
    submit,
    tags,
)

api_router = APIRouter()
api_router.include_router(
    artifacts.router, tags=["artifacts"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    background_tasks.router,
    tags=["background-tasks"],
    dependencies=[Depends(deps.AuthVerifier)],
)
api_router.include_router(
    files.router, tags=["files"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    functions.router, tags=["functions"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(healthz.router, tags=["healthz"])
api_router.include_router(
    logs.router, tags=["logs"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    pipelines.router, tags=["pipelines"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    projects.router, tags=["projects"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    runs.router, tags=["runs"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    runtimes.router, tags=["runtimes"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    schedules.router, tags=["schedules"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    submit.router, tags=["submit"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    tags.router, tags=["tags"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(
    feature_store.router,
    tags=["feature-store"],
    dependencies=[Depends(deps.AuthVerifier)],
)
api_router.include_router(
    frontend_spec.router,
    tags=["frontend-specs"],
    dependencies=[Depends(deps.AuthVerifier)],
)
api_router.include_router(
    secrets.router, tags=["secrets"], dependencies=[Depends(deps.AuthVerifier)]
)
api_router.include_router(grafana_proxy.router, tags=["grafana", "model_endpoints"])
api_router.include_router(model_endpoints.router, tags=["model_endpoints"])
