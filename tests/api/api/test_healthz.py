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
    overridden_ui_projects_prefix = "some-prefix"
    mlrun.mlconf.ui.projects_prefix = overridden_ui_projects_prefix
    nuclio_version = "x.x.x"
    mlrun.mlconf.nuclio_version = nuclio_version
    response = client.get("healthz")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    for key in ["scrape_metrics", "hub_url"]:
        assert response_body[key] is None
    assert response_body["ui_projects_prefix"] == overridden_ui_projects_prefix
    assert response_body["nuclio_version"] == nuclio_version
