import datetime
import http
import json

import deepdiff
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
            "urls": [
                {"kind": "http", "url": "https-has-precedence"},
                {"kind": "https", "url": expected_grafana_url},
            ],
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


def test_create_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(
            name="project-name",
            created=datetime.datetime.utcnow(),
            labels={"some-label": "some-label-value",},
            annotations={"some-annotation": "some-annotation-value",},
            some_extra_field="some value",
        ),
        spec=mlrun.api.schemas.ProjectSpec(
            description="project description",
            desired_state=mlrun.api.schemas.ProjectState.online,
            some_extra_field="some value",
        ),
        status=mlrun.api.schemas.ProjectStatus(some_extra_field="some value",),
    )

    def verify_creation(request, context):
        _assert_project_creation(iguazio_client, request.json(), project)
        context.status_code = http.HTTPStatus.CREATED.value
        return _build_project_response(iguazio_client, project)

    requests_mock.post(f"{api_url}/api/projects", json=verify_creation)
    created_project = iguazio_client.create_project(None, project,)
    exclude = {"status": {"state"}}
    assert (
        deepdiff.DeepDiff(
            project.dict(exclude=exclude),
            created_project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert created_project.status.state == project.spec.desired_state


def _build_project_response(
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    project: mlrun.api.schemas.Project,
):
    body = {
        "data": {
            "type": "project",
            "attributes": {
                "name": project.metadata.name,
                "created_at": project.metadata.created.isoformat()
                if project.metadata.created
                else datetime.datetime.utcnow().isoformat(),
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "admin_status": project.spec.desired_state
                or mlrun.api.schemas.ProjectState.online,
                "mlrun_project": iguazio_client._transform_mlrun_project_to_iguazio_mlrun_project_attribute(
                    project
                ),
            },
        }
    }
    if project.spec.description:
        body["data"]["attributes"]["description"] = project.spec.description
    if project.metadata.labels:
        body["data"]["attributes"][
            "labels"
        ] = iguazio_client._transform_mlrun_labels_to_iguazio_labels(
            project.metadata.labels
        )
    if project.metadata.annotations:
        body["data"]["attributes"][
            "annotations"
        ] = iguazio_client._transform_mlrun_labels_to_iguazio_labels(
            project.metadata.annotations
        )
    body["data"]["attributes"]["operational_status"] = body["data"]["attributes"][
        "admin_status"
    ]
    return body


def _assert_project_creation(
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    request_body: dict,
    project: mlrun.api.schemas.Project,
):
    assert request_body["data"]["attributes"]["name"] == project.metadata.name
    assert request_body["data"]["attributes"]["description"] == project.spec.description
    assert (
        request_body["data"]["attributes"]["admin_status"] == project.spec.desired_state
    )
    assert request_body["data"]["attributes"]["mlrun_project"] == json.dumps(
        project.dict(
            exclude_unset=True,
            exclude={
                "metadata": {"name", "created", "labels", "annotations"},
                "spec": {"description", "desired_state"},
                "status": {"state"},
            },
        )
    )
    if project.metadata.created:
        assert (
            request_body["data"]["attributes"]["created_at"]
            == project.metadata.created.isoformat()
        )
    if project.metadata.labels:
        assert request_body["data"]["attributes"][
            "labels"
        ] == iguazio_client._transform_mlrun_labels_to_iguazio_labels(
            project.metadata.labels
        )
    if project.metadata.annotations:
        assert request_body["data"]["attributes"][
            "annotations"
        ] == iguazio_client._transform_mlrun_labels_to_iguazio_labels(
            project.metadata.annotations
        )


def _generate_app_services_manifests_body(app_services):
    return {"data": [{"attributes": {"app_services": app_services}}]}
