import http

import fastapi.testclient
import sqlalchemy.orm

import mlrun
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors
import mlrun.runtimes


def test_health(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    overridden_ui_projects_prefix = 'some-prefix'
    mlrun.mlconf.ui.projects_prefix = overridden_ui_projects_prefix
    response = client.get("/api/healthz")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    for key in ['scrape_metrics']:
        assert response_body[key] is None
    assert response_body['ui_projects_prefix'] == overridden_ui_projects_prefix
