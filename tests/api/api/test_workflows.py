import deepdiff
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_list_workflows(db: Session, client: TestClient) -> None:
    response = client.get("/api/workflows")
    assert response.status_code == HTTPStatus.OK.value
    expected_response = {
        "runs": [],
        "total_size": 0,
        "next_page_token": None,
    }
    assert (
        deepdiff.DeepDiff(expected_response, response.json(), ignore_order=True,) == {}
    )
