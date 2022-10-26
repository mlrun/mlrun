from fastapi import APIRouter, Depends

import mlrun.api.api.deps
from . import memory_reports


internal_router = APIRouter(
    prefix="/_internal", dependencies=[Depends(mlrun.api.api.deps.verify_api_state)], tags=["internal"],
)

internal_router.include_router(
    memory_reports.router,
    tags=["memory-reports"],
    dependencies=[Depends(mlrun.api.api.deps.authenticate_request)],
)
