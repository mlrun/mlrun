from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_list_schedules(db: Session, client: TestClient) -> None:
    resp = client.get('/api/projects/default/schedules')
    assert resp.status_code == status.HTTP_200_OK, 'status'
    assert 'schedules' in resp.json(), 'no schedules'
