import http
import unittest.mock

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors


def test_get_frontend_spec(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    grafana_url = "some-url.com"
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url = unittest.mock.Mock(
        return_value=grafana_url
    )

    # no cookie so no url
    response = client.get("/api/frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert frontend_spec.jobs_dashboard_url is None
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url.assert_not_called()

    response = client.get(
        "/api/frontend-spec", cookies={"session": "some-session-cookie"}
    )
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert (
        frontend_spec.jobs_dashboard_url
        == f"{grafana_url}/d/mlrun-jobs-monitoring/mlrun-jobs-monitoring?orgId=1"
        f"&var-groupBy={{filter_name}}&var-filter={{filter_value}}"
    )
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url.assert_called_once()
