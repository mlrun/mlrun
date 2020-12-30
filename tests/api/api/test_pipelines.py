from http import HTTPStatus

import deepdiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_list_pipelines_not_exploding_on_no_k8s(db: Session, client: TestClient) -> None:
    response = client.get("/api/projects/*/pipelines")
    assert response.status_code == HTTPStatus.OK.value
    expected_response = {
        "runs": [],
        "total_size": 0,
        "next_page_token": None,
    }
    assert (
        deepdiff.DeepDiff(expected_response, response.json(), ignore_order=True,) == {}
    )
