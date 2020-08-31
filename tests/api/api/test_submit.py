from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_submit_job_failure_function_not_found(db: Session, client: TestClient) -> None:
    function_reference = (
        "cat-and-dog-servers/aggregate@b145b6d958a7b4d84f12821a06459e31ea422308"
    )
    body = {
        "task": {"spec": {"function": function_reference}},
    }
    resp = client.post("/api/submit_job", json=body)
    assert resp.status_code == HTTPStatus.NOT_FOUND.value
    assert f"Function not found {function_reference}" in resp.json()["detail"]
