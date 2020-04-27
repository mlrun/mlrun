from fastapi import APIRouter

from mlrun.app.api.endpoints import artifacts, functions, logs, pipelines, projects, runs, schedules

api_router = APIRouter()
api_router.include_router(artifacts.router, tags=["artifacts"])
api_router.include_router(functions.router, tags=["functions"])
api_router.include_router(logs.router, tags=["logs"])
api_router.include_router(pipelines.router, tags=["pipelines"])
api_router.include_router(projects.router, tags=["projects"])
api_router.include_router(runs.router, tags=["runs"])
api_router.include_router(schedules.router, tags=["schedules"])
