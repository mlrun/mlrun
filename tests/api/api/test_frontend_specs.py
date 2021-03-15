import unittest.mock
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors


def test_get_frontend_specs(db: Session, client: TestClient) -> None:
    grafana_url = "some-url.com"
    mlrun.api.utils.clients.iguazio.Client().get_grafana_service_url_if_exists = unittest.mock.Mock(
        return_value=grafana_url
    )

    # no cookie so no url
    response = client.get("/api/frontend-specs")
    assert response.status_code == HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert frontend_spec.jobs_dashboard_url is None
    assert (
        mlrun.api.utils.clients.iguazio.Client().get_grafana_service_url_if_exists.call_count
        == 0
    )

    response = client.get(
        "/api/frontend-specs", cookies={"session": "some-session-cookie"}
    )
    assert response.status_code == HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert (
        frontend_spec.jobs_dashboard_url
        == f"{grafana_url}/d/mlrun-jobs-monitoring/mlrun-jobs-monitoring?orgId=1"
        f"&var-groupBy={{filter_name}}&var-filter={{filter_value}}"
    )
    assert (
        mlrun.api.utils.clients.iguazio.Client().get_grafana_service_url_if_exists.call_count
        == 1
    )
