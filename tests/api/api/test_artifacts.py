from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas

PROJECT = "prj"
KEY = "some-key"
UID = "some-uid"
TAG = "some-tag"
API_PROJECTS_PATH = "/api/projects"
API_ARTIFACT_PATH = "/api/artifact"
API_ARTIFACTS_PATH = "/api/artifacts"


def test_list_artifact_tags(db: Session, client: TestClient) -> None:
    resp = client.get(f"{API_PROJECTS_PATH}/{PROJECT}/artifact-tags")
    assert resp.status_code == HTTPStatus.OK, "status"
    assert resp.json()["project"] == PROJECT, "project"


def _create_project(client: TestClient, project_name: str = PROJECT):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = client.post(API_PROJECTS_PATH, json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED
    return resp


def test_store_artifact_with_empty_dict(db: Session, client: TestClient):
    _create_project(client)

    resp = client.post(
        f"{API_ARTIFACT_PATH}/{PROJECT}/{UID}/{KEY}?tag={TAG}", data="{}"
    )
    assert resp.status_code == HTTPStatus.OK

    resp = client.get(f"{API_PROJECTS_PATH}/{PROJECT}/artifact/{KEY}?tag={TAG}")
    assert resp.status_code == HTTPStatus.OK


def test_delete_artifacts_after_storing_empty_dict(db: Session, client: TestClient):
    _create_project(client)
    empty_artifact = "{}"

    resp = client.post(
        f"{API_ARTIFACT_PATH}/{PROJECT}/{UID}/{KEY}?tag={TAG}", data=empty_artifact
    )
    assert resp.status_code == HTTPStatus.OK

    resp = client.post(
        f"{API_ARTIFACT_PATH}/{PROJECT}/{UID}2/{KEY}2?tag={TAG}", data=empty_artifact
    )
    assert resp.status_code == HTTPStatus.OK

    project_artifacts_path = f"{API_ARTIFACTS_PATH}?project={PROJECT}"

    resp = client.get(project_artifacts_path)
    assert len(resp.json()["artifacts"]) == 2

    resp = client.delete(project_artifacts_path)
    assert resp.status_code == HTTPStatus.OK

    resp = client.get(project_artifacts_path)
    assert len(resp.json()["artifacts"]) == 0
