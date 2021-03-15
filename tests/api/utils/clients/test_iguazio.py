import pytest
import requests_mock as requests_mock_package

import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.config
import mlrun.errors


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://iguazio-api-url:8080"
    mlrun.config.config._iguazio_api_url = api_url
    return api_url


@pytest.fixture()
async def iguazio_client(api_url: str,) -> mlrun.api.utils.clients.iguazio.Client:
    client = mlrun.api.utils.clients.iguazio.Client()
    # force running init again so the configured api url will be used
    client.__init__()
    return client


def test_get_grafana_service_url_success(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    expected_grafana_url = (
        "https://grafana.default-tenant.app.hedingber-301-1.iguazio-cd2.com"
    )
    grafana_service = {
        "spec": {"kind": "grafana"},
        "status": {
            "state": "ready",
            "urls": [{"kind": "https", "url": expected_grafana_url}],
        },
    }
    response_body = _generate_app_services_manifests_body([grafana_service])
    requests_mock.get(f"{api_url}/api/app_services_manifests", json=response_body)
    grafana_url = iguazio_client.try_get_grafana_service_url("session-cookie")
    assert grafana_url == expected_grafana_url


def test_get_grafana_service_url_ignoring_disabled_service(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    grafana_service = {"spec": {"kind": "grafana"}, "status": {"state": "disabled"}}
    response_body = _generate_app_services_manifests_body([grafana_service])
    requests_mock.get(f"{api_url}/api/app_services_manifests", json=response_body)
    grafana_url = iguazio_client.try_get_grafana_service_url("session-cookie")
    assert grafana_url is None


def test_get_grafana_service_url_no_grafana_exists(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    response_body = _generate_app_services_manifests_body([])
    requests_mock.get(f"{api_url}/api/app_services_manifests", json=response_body)
    grafana_url = iguazio_client.try_get_grafana_service_url("session-cookie")
    assert grafana_url is None


def test_get_grafana_service_url_no_urls(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    grafana_service = {
        "spec": {"kind": "grafana"},
        "status": {"state": "ready", "urls": []},
    }
    response_body = _generate_app_services_manifests_body([grafana_service])
    requests_mock.get(f"{api_url}/api/app_services_manifests", json=response_body)
    grafana_url = iguazio_client.try_get_grafana_service_url("session-cookie")
    assert grafana_url is None


def _generate_app_services_manifests_body(app_services):
    return {"data": [{"attributes": {"app_services": app_services}}]}
