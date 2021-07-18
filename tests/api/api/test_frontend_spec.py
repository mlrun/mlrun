import http
import unittest.mock

import deepdiff
import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors
import mlrun.runtimes


def test_get_frontend_spec(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url = (
        unittest.mock.Mock()
    )
    response = client.get("/api/frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert (
        deepdiff.DeepDiff(
            frontend_spec.abortable_function_kinds,
            mlrun.runtimes.RuntimeKinds.abortable_runtimes(),
        )
        == {}
    )
    assert frontend_spec.feature_flags.project_membership == mlrun.api.schemas.ProjectMembershipFeatureFlag.disabled


def test_get_frontend_spec_jobs_dashboard_url_resolution(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url = (
        unittest.mock.Mock()
    )
    # no cookie so no url
    response = client.get("/api/frontend-spec")
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert frontend_spec.jobs_dashboard_url is None
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url.assert_not_called()

    # no grafana (None returned) so no url
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url = unittest.mock.Mock(
        return_value=None
    )
    response = client.get(
        "/api/frontend-spec", cookies={"session": "some-session-cookie"}
    )
    assert response.status_code == http.HTTPStatus.OK.value
    frontend_spec = mlrun.api.schemas.FrontendSpec(**response.json())
    assert frontend_spec.jobs_dashboard_url is None
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url.assert_called_once()

    # happy secnario - grafana url found, verify returned correctly
    grafana_url = "some-url.com"
    mlrun.api.utils.clients.iguazio.Client().try_get_grafana_service_url = unittest.mock.Mock(
        return_value=grafana_url
    )

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
