from fastapi import FastAPI

from mlrun.app.api.api import api_router

app = FastAPI()

app.include_router(api_router, prefix='/api')
