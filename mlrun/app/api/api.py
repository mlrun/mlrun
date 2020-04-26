from fastapi import APIRouter

from mlrun.app.api.endpoints import artifacts, functions, runs, schedules

api_router = APIRouter()
api_router.include_router(artifacts.router, tags=["artifacts"])
api_router.include_router(functions.router, tags=["functions"])
api_router.include_router(runs.router, tags=["runs"])
api_router.include_router(schedules.router, tags=["schedules"])
