from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_list_schedules(db: Session, client: TestClient) -> None:
    resp = client.get('/api/schedules')
    assert resp.status_code == HTTPStatus.OK, 'status'
    assert 'schedules' in resp.json(), 'no schedules'
