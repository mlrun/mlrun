from fastapi import APIRouter, Depends

from mlrun.api.api import deps
from mlrun.api.api.endpoints import (
    artifacts,
    background_tasks,
    files,
    functions,
    healthz,
    logs,
    pipelines,
    projects,
    runs,
    runtimes,
    schedules,
    submit,
    tags,
    feature_sets,
    model_endpoints,
    secrets,
    grafana_proxy,
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
    feature_sets.router,
    tags=["feature-sets"],
    dependencies=[Depends(deps.AuthVerifier)],
)
api_router.include_router(
    secrets.router, tags=["secrets"], dependencies=[Depends(deps.AuthVerifier)]
)
# Since grafana-proxy imitates the path schema that was set for model-endpoints, ordering between the following two
# routers is crucial for functionality of the grafana-proxy router.
# ** Make sure the inclusion of the grafana_proxy.router happens before the inclusion of the model_endpoints.router. **
api_router.include_router(grafana_proxy.router, tags=["grafana", "model_endpoints"])
api_router.include_router(model_endpoints.router, tags=["model_endpoints"])
