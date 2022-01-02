import http
import os

import fastapi.testclient
import pytest
import sqlalchemy.orm


def test_docs(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == http.HTTPStatus.OK.value


@pytest.mark.skipif(
    os.getenv("MLRUN_OPENAPI_JSON_TARGET_PATH") is None,
    reason="Supposed to run only for CI backward compatibility tests",
)
def test_save_openapi_json(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    """"The purpose of the test is to create an openapi.json file that is used to run backward compatibility tests"""
    response = client.get("/api/openapi.json")
    with open(
        os.path.join(os.getenv("MLRUN_OPENAPI_JSON_TARGET_PATH"), "openapi.json"), "w"
    ) as file:
        file.write(response.text)
