import http

import fastapi.testclient
import sqlalchemy.orm


def test_docs(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == http.HTTPStatus.OK.value
