import datetime
import functools
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
    client._wait_for_job_completion_retry_interval = 0
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


def test_list_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    mock_projects = [
        {"name": "project-name-1"},
        {"name": "project-name-2", "description": "project-description-2"},
        {"name": "project-name-3", "labels": {"key": "value"}},
        {
            "name": "project-name-4",
            "annotations": {"annotation-key": "annotation-value"},
        },
        {
            "name": "project-name-5",
            "description": "project-description-4",
            "labels": {"key2": "value2"},
            "annotations": {"annotation-key2": "annotation-value2"},
        },
    ]
    response_body = {
        "data": [
            _build_project_response(
                iguazio_client,
                _generate_project(
                    mock_project["name"],
                    mock_project.get("description", ""),
                    mock_project.get("labels", {}),
                    mock_project.get("annotations", {}),
                ),
            )
            for mock_project in mock_projects
        ]
    }
    requests_mock.get(f"{api_url}/api/projects", json=response_body)
    projects = iguazio_client.list_projects(None)
    for index, project in enumerate(projects):
        assert project.metadata.name == mock_projects[index]["name"]
        assert project.spec.description == mock_projects[index].get("description")
        assert (
            deepdiff.DeepDiff(
                mock_projects[index].get("labels"),
                project.metadata.labels,
                ignore_order=True,
            )
            == {}
        )
        assert (
            deepdiff.DeepDiff(
                mock_projects[index].get("annotations"),
                project.metadata.annotations,
                ignore_order=True,
            )
            == {}
        )


def test_create_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"

    def verify_creation(request, context):
        _assert_project_creation(iguazio_client, request.json(), project)
        context.status_code = http.HTTPStatus.CREATED.value
        assert request.headers["Cookie"] == f"session={session_cookie}"
        return {"data": _build_project_response(iguazio_client, project)}

    requests_mock.post(f"{api_url}/api/projects", json=verify_creation)
    created_project = iguazio_client.create_project(session_cookie, project,)
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


def test_store_project_creation(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"

    def verify_store_creation(request, context):
        _assert_project_creation(iguazio_client, request.json(), project)
        context.status_code = http.HTTPStatus.CREATED.value
        assert request.headers["Cookie"] == f"session={session_cookie}"
        return {"data": _build_project_response(iguazio_client, project)}

    # mock project not found so store will create
    requests_mock.get(
        f"{api_url}/api/projects/{project.metadata.name}",
        status_code=http.HTTPStatus.NOT_FOUND.value,
    )
    requests_mock.post(f"{api_url}/api/projects", json=verify_store_creation)
    created_project = iguazio_client.store_project(
        session_cookie, project.metadata.name, project
    )
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


def test_store_project_update(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"

    def verify_store_update(request, context):
        _assert_project_creation(iguazio_client, request.json(), project)
        context.status_code = http.HTTPStatus.OK.value
        assert request.headers["Cookie"] == f"session={session_cookie}"
        return {"data": _build_project_response(iguazio_client, project)}

    empty_project = _generate_project(description="", labels={}, annotations={})
    # mock project response so store will update
    requests_mock.get(
        f"{api_url}/api/projects/{project.metadata.name}",
        json=_build_project_response(iguazio_client, empty_project),
    )
    requests_mock.put(
        f"{api_url}/api/projects/{project.metadata.name}", json=verify_store_update
    )
    updated_project = iguazio_client.store_project(
        session_cookie, project.metadata.name, project,
    )
    exclude = {"status": {"state"}}
    assert (
        deepdiff.DeepDiff(
            project.dict(exclude=exclude),
            updated_project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert updated_project.status.state == project.spec.desired_state


def test_delete_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    job_id = "928145d5-4037-40b0-98b6-19a76626d797"
    session_cookie = "1234"

    def verify_deletion(request, context):
        assert request.json()["data"]["attributes"]["name"] == project_name
        assert (
            request.headers["x-iguazio-delete-project-strategy"]
            == mlrun.api.schemas.DeletionStrategy.default().to_nuclio_deletion_strategy()
        )
        assert request.headers["Cookie"] == f"session={session_cookie}"
        context.status_code = http.HTTPStatus.ACCEPTED.value
        return {"data": {"type": "job", "id": job_id}}

    def mock_get_job_in_progress(state, request, context):
        context.status_code = http.HTTPStatus.OK.value
        assert request.headers["Cookie"] == f"session={session_cookie}"
        return {"data": {"attributes": {"state": state}}}

    requests_mock.delete(f"{api_url}/api/projects", json=verify_deletion)
    responses = [
        functools.partial(mock_get_job_in_progress, "in_progress"),
        functools.partial(mock_get_job_in_progress, "in_progress"),
        functools.partial(mock_get_job_in_progress, "completed"),
    ]
    mocker = requests_mock.get(
        f"{api_url}/api/jobs/{job_id}",
        response_list=[{"json": response} for response in responses],
    )
    iguazio_client.delete_project(session_cookie, project_name)
    assert mocker.call_count == len(responses)

    # assert ignoring (and not exploding) on not found
    requests_mock.delete(
        f"{api_url}/api/projects", status_code=http.HTTPStatus.NOT_FOUND.value
    )
    iguazio_client.delete_project(session_cookie, project_name)

    # TODO: not sure really needed
    # assert correctly propagating 412 errors (will be returned when project has resources)
    requests_mock.delete(
        f"{api_url}/api/projects", status_code=http.HTTPStatus.PRECONDITION_FAILED.value
    )
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        iguazio_client.delete_project(session_cookie, project_name)


def _generate_project(
    name="project-name",
    description="project description",
    labels=None,
    annotations=None,
) -> mlrun.api.schemas.Project:
    if labels is None:
        labels = {
            "some-label": "some-label-value",
        }
    if annotations is None:
        annotations = {
            "some-annotation": "some-annotation-value",
        }
    return mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(
            name=name,
            created=datetime.datetime.utcnow(),
            labels=labels,
            annotations=annotations,
            some_extra_field="some value",
        ),
        spec=mlrun.api.schemas.ProjectSpec(
            description=description,
            desired_state=mlrun.api.schemas.ProjectState.online,
            some_extra_field="some value",
        ),
        status=mlrun.api.schemas.ProjectStatus(some_extra_field="some value",),
    )


def _build_project_response(
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    project: mlrun.api.schemas.Project,
):
    body = {
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
    if project.spec.description:
        body["attributes"]["description"] = project.spec.description
    if project.metadata.labels:
        body["attributes"][
            "labels"
        ] = iguazio_client._transform_mlrun_labels_to_iguazio_labels(
            project.metadata.labels
        )
    if project.metadata.annotations:
        body["attributes"][
            "annotations"
        ] = iguazio_client._transform_mlrun_labels_to_iguazio_labels(
            project.metadata.annotations
        )
    body["attributes"]["operational_status"] = body["attributes"]["admin_status"]
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
