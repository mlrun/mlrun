from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_list_artifact_tags(db: Session, client: TestClient) -> None:
    project = "p11"
    resp = client.get(f"/api/projects/{project}/artifact-tags")
    assert resp.status_code == HTTPStatus.OK.value, "status"
    assert resp.json()["project"] == project, "project"
