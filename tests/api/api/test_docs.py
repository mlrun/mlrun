import http

import fastapi.testclient
import sqlalchemy.orm
import os
import pytest


def test_docs(
        db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == http.HTTPStatus.OK.value


def does_env_exists(env):
    if os.getenv(env):
        return False
    return True


@pytest.mark.skipif(does_env_exists('OPENAPI_JSON_TARGET_PATH'),
                    "Supposed to run only for CI backward compatibility checks")
def test_save_openapi_json(db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
                           ) -> None:
    response = client.get("/api/openapi.json")
    with open(os.path.join(os.getenv('OPENAPI_JSON_TARGET_PATH'), 'openapi.json'), "w") as openapi:
        openapi.write(response.text)
