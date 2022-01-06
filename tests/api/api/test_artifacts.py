from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas

PROJECT = "prj"
KEY = "some-key"
UID = "some-uid"
TAG = "some-tag"


def test_list_artifact_tags(db: Session, client: TestClient) -> None:
    resp = client.get(f"/api/projects/{PROJECT}/artifact-tags")
    assert resp.status_code == HTTPStatus.OK, "status"
    assert resp.json()["PROJECT"] == PROJECT, "PROJECT"


def _create_project(client: TestClient, project_name: str = PROJECT):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = client.post("/api/projects", json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED
    return resp


def test_store_artifact_with_empty_dict(db: Session, client: TestClient):
    _create_project(client)

    resp = client.post(f"/api/artifact/{PROJECT}/{UID}/{KEY}?tag={TAG}", data="{}")
    assert resp.status_code == HTTPStatus.OK

    resp = client.get(f"/api/projects/{PROJECT}/artifact/{KEY}?tag={TAG}")
    assert resp.status_code == HTTPStatus.OK


def test_delete_artifacts_after_storing_empty_dict(db: Session, client: TestClient):
    _create_project(client)
    artifacts_path = "/api/artifacts"
    empty_artifact = "{}"

    resp = client.post(
        f"/api/artifact/{PROJECT}/{UID}/{KEY}?tag={TAG}", data=empty_artifact
    )
    assert resp.status_code == HTTPStatus.OK

    uid2 = "uid2"
    key2 = "key2"
    resp = client.post(
        f"/api/artifact/{PROJECT}/{uid2}/{key2}?tag={TAG}", data=empty_artifact
    )
    assert resp.status_code == HTTPStatus.OK

    project_artifacts_path = f"{artifacts_path}?project={PROJECT}"

    resp = client.get(project_artifacts_path)
    assert len(resp.json()["artifacts"]) == 2

    resp = client.delete(project_artifacts_path)
    assert resp.status_code == HTTPStatus.OK

    resp = client.get(project_artifacts_path)
    assert len(resp.json()["artifacts"]) == 0


