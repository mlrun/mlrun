from fastapi import APIRouter

from mlrun.app.api.endpoints import artifacts, files, functions, healthz, logs, pipelines, projects, runs, schedules, \
    submit, tags, workflows

api_router = APIRouter()
api_router.include_router(artifacts.router, tags=["artifacts"])
api_router.include_router(files.router, tags=["files"])
api_router.include_router(functions.router, tags=["functions"])
api_router.include_router(healthz.router, tags=["healthz"])
api_router.include_router(logs.router, tags=["logs"])
api_router.include_router(pipelines.router, tags=["pipelines"])
api_router.include_router(projects.router, tags=["projects"])
api_router.include_router(runs.router, tags=["runs"])
api_router.include_router(schedules.router, tags=["schedules"])
api_router.include_router(submit.router, tags=["submit"])
api_router.include_router(tags.router, tags=["tags"])
api_router.include_router(workflows.router, tags=["workflows"])
