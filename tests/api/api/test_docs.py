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


def does_env_exists(env):
    if os.getenv(env):
        return True
    return False


@pytest.mark.skipif(
    os.getenv("OPENAPI_JSON_TARGET_PATH") is None,
    reason="Supposed to run only for CI backward compatibility checks",
)
def test_save_openapi_json(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("/api/openapi.json")
    with open(
        os.path.join(os.getenv("OPENAPI_JSON_TARGET_PATH"), "openapi.json"), "w"
    ) as openapi:
        openapi.write(response.text)
