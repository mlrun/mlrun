from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_list_artifact_tags(db: Session, client: TestClient) -> None:
    project = 'p11'
    resp = client.get(f'/api/projects/{project}/artifact-tags')
    assert resp.status_code == status.HTTP_200_OK, 'status'
    assert resp.json()['project'] == project, 'project'
