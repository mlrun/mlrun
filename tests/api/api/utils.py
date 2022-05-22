from http import HTTPStatus

import httpx
from fastapi.testclient import TestClient

import mlrun.api.api.endpoints.functions
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.errors

PROJECT = "project-name"


def create_project(client: TestClient, project_name: str = PROJECT):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = client.post("projects", json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


async def create_project_async(
    async_client: httpx.AsyncClient, project_name: str = PROJECT
):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = await async_client.post(
        "projects",
        json=project.dict(),
    )
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp
