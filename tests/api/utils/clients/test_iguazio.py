import datetime
import functools
import http
import json
import typing

import deepdiff
import fastapi
import pytest
import requests_mock as requests_mock_package
import starlette.datastructures

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
    client._wait_for_project_terminal_state_retry_interval = 0
    return client


def test_verify_request_session_success(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    mock_request_headers = starlette.datastructures.Headers(
        {"cookie": "session=some-session-cookie"}
    )
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = mock_request_headers

    mock_response_headers = {
        "X-Remote-User": "some-user",
        "X-V3io-Session-Key": "some-access-key",
        "X-uid": "some-uid",
        "X-gids": "some-gid,some-gid2,some-gid3",
    }

    def _verify_session_mock(request, context):
        for header_key, header_value in mock_request_headers.items():
            assert request.headers[header_key] == header_value
        context.headers = mock_response_headers
        return {}

    requests_mock.post(
        f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}",
        json=_verify_session_mock,
    )
    (username, access_key, uid, gids,) = iguazio_client.verify_request_session(
        mock_request
    )
    assert username == mock_response_headers["X-Remote-User"]
    assert access_key == mock_response_headers["X-V3io-Session-Key"]
    assert uid == mock_response_headers["X-uid"]
    assert gids == mock_response_headers["X-gids"].split(",")


def test_verify_request_session_failure(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers()

    requests_mock.post(
        f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}",
        status_code=http.HTTPStatus.UNAUTHORIZED.value,
    )
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        iguazio_client.verify_request_session(mock_request)


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


def test_list_project_with_updated_after(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"
    updated_after = datetime.datetime.now(tz=datetime.timezone.utc)

    def verify_list(request, context):
        assert request.qs == {
            "filter[updated_at]": [
                f"[$gt]{updated_after.isoformat().split('+')[0]}Z".lower()
            ]
        }
        context.status_code = http.HTTPStatus.OK.value
        _verify_request_headers(request.headers, session_cookie)
        return {"data": [_build_project_response(iguazio_client, project)]}

    # mock project response so store will update
    requests_mock.get(
        f"{api_url}/api/projects", json=verify_list,
    )
    iguazio_client.list_projects(
        session_cookie, updated_after,
    )


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
    projects, latest_updated_at = iguazio_client.list_projects(None)
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
    assert (
        latest_updated_at.isoformat()
        == response_body["data"][-1]["attributes"]["updated_at"]
    )


def test_create_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    _create_project_and_assert(api_url, iguazio_client, requests_mock, project)


def test_create_project_minimal_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name="some-name",),
    )
    _create_project_and_assert(api_url, iguazio_client, requests_mock, project)


def test_create_project_without_wait(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session_cookie, job_id
        ),
    )
    created_project, is_running_in_background = iguazio_client.create_project(
        session_cookie, project, wait_for_completion=False
    )
    assert is_running_in_background is True
    exclude = {"status": {"state"}}
    assert (
        deepdiff.DeepDiff(
            project.dict(exclude=exclude),
            created_project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert created_project.status.state == mlrun.api.schemas.ProjectState.creating


def test_store_project_creation(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    # mock project not found so store will create - then successful response which used to get the created project
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        response_list=[
            {"status_code": http.HTTPStatus.NOT_FOUND.value},
            {"json": {"data": _build_project_response(iguazio_client, project)}},
        ],
    )
    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session_cookie, job_id
        ),
    )
    mocker, num_of_calls_until_completion = _mock_job_progress(
        api_url, requests_mock, session_cookie, job_id
    )
    created_project, is_running_in_background = iguazio_client.store_project(
        session_cookie, project.metadata.name, project
    )
    assert is_running_in_background is False
    assert mocker.call_count == num_of_calls_until_completion
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


def test_store_project_creation_without_wait(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    # mock project not found so store will create - then successful response which used to get the created project
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        response_list=[
            {"status_code": http.HTTPStatus.NOT_FOUND.value},
            {"json": {"data": _build_project_response(iguazio_client, project)}},
        ],
    )
    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session_cookie, job_id
        ),
    )
    created_project, is_running_in_background = iguazio_client.store_project(
        session_cookie, project.metadata.name, project, wait_for_completion=False
    )
    assert is_running_in_background is True
    exclude = {"status": {"state"}}
    assert (
        deepdiff.DeepDiff(
            project.dict(exclude=exclude),
            created_project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert created_project.status.state == mlrun.api.schemas.ProjectState.creating


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
        _verify_request_headers(request.headers, session_cookie)
        return {"data": _build_project_response(iguazio_client, project)}

    empty_project = _generate_project(description="", labels={}, annotations={})
    # mock project response so store will update
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json={"data": _build_project_response(iguazio_client, empty_project)},
    )
    requests_mock.put(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json=verify_store_update,
    )
    updated_project, is_running_in_background = iguazio_client.store_project(
        session_cookie, project.metadata.name, project,
    )
    assert is_running_in_background is False
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


def test_patch_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session_cookie = "1234"
    patched_description = "new desc"

    def verify_patch(request, context):
        patched_project = _generate_project(
            description=patched_description, created=project.metadata.created
        )
        _assert_project_creation(iguazio_client, request.json(), patched_project)
        context.status_code = http.HTTPStatus.OK.value
        _verify_request_headers(request.headers, session_cookie)
        return {"data": _build_project_response(iguazio_client, patched_project)}

    # mock project response on get (patch does get first)
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json={"data": _build_project_response(iguazio_client, project)},
    )
    requests_mock.put(
        f"{api_url}/api/projects/__name__/{project.metadata.name}", json=verify_patch,
    )
    patched_project, is_running_in_background = iguazio_client.patch_project(
        session_cookie,
        project.metadata.name,
        {"spec": {"description": patched_description}},
        wait_for_completion=True,
    )
    assert is_running_in_background is False
    exclude = {"status": {"state"}, "spec": {"description"}}
    assert (
        deepdiff.DeepDiff(
            project.dict(exclude=exclude),
            patched_project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert patched_project.status.state == project.spec.desired_state
    assert patched_project.spec.description == patched_description


def test_delete_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    job_id = "928145d5-4037-40b0-98b6-19a76626d797"
    session_cookie = "1234"

    requests_mock.delete(
        f"{api_url}/api/projects",
        json=functools.partial(_verify_deletion, project_name, session_cookie, job_id),
    )
    mocker, num_of_calls_until_completion = _mock_job_progress(
        api_url, requests_mock, session_cookie, job_id
    )
    is_running_in_background = iguazio_client.delete_project(
        session_cookie, project_name
    )
    assert is_running_in_background is False
    assert mocker.call_count == num_of_calls_until_completion

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


def test_delete_project_without_wait(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    job_id = "928145d5-4037-40b0-98b6-19a76626d797"
    session_cookie = "1234"

    requests_mock.delete(
        f"{api_url}/api/projects",
        json=functools.partial(_verify_deletion, project_name, session_cookie, job_id),
    )
    is_running_in_background = iguazio_client.delete_project(
        session_cookie, project_name, wait_for_completion=False
    )
    assert is_running_in_background is True


def _create_project_and_assert(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
    project: mlrun.api.schemas.Project,
):
    session_cookie = "1234"
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session_cookie, job_id
        ),
    )
    mocker, num_of_calls_until_completion = _mock_job_progress(
        api_url, requests_mock, session_cookie, job_id
    )
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json={"data": _build_project_response(iguazio_client, project)},
    )
    created_project, is_running_in_background = iguazio_client.create_project(
        session_cookie, project,
    )
    assert is_running_in_background is False
    assert mocker.call_count == num_of_calls_until_completion
    exclude = {"metadata": {"created"}, "status": {"state"}}
    assert (
        deepdiff.DeepDiff(
            project.dict(exclude=exclude),
            created_project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert created_project.metadata.created is not None
    assert created_project.status.state == project.spec.desired_state


def _verify_deletion(project_name, session_cookie, job_id, request, context):
    assert request.json()["data"]["attributes"]["name"] == project_name
    assert (
        request.headers["igz-project-deletion-strategy"]
        == mlrun.api.schemas.DeletionStrategy.default().to_iguazio_deletion_strategy()
    )
    _verify_request_headers(request.headers, session_cookie)
    context.status_code = http.HTTPStatus.ACCEPTED.value
    return {"data": {"type": "job", "id": job_id}}


def _verify_creation(iguazio_client, project, session_cookie, job_id, request, context):
    _assert_project_creation(iguazio_client, request.json(), project)
    context.status_code = http.HTTPStatus.CREATED.value
    _verify_request_headers(request.headers, session_cookie)
    return {
        "data": _build_project_response(
            iguazio_client, project, job_id, mlrun.api.schemas.ProjectState.creating
        )
    }


def _verify_request_headers(headers: dict, session_cookie: str):
    assert headers["Cookie"] == f"session={session_cookie}"
    assert headers[mlrun.api.schemas.HeaderNames.projects_role] == "mlrun"


def _mock_job_progress(api_url, requests_mock, session_cookie: str, job_id: str):
    def _mock_get_job(state, session_cookie, request, context):
        context.status_code = http.HTTPStatus.OK.value
        assert request.headers["Cookie"] == f"session={session_cookie}"
        return {"data": {"attributes": {"state": state}}}

    responses = [
        functools.partial(_mock_get_job, "in_progress", session_cookie),
        functools.partial(_mock_get_job, "in_progress", session_cookie),
        functools.partial(_mock_get_job, "completed", session_cookie),
    ]
    mocker = requests_mock.get(
        f"{api_url}/api/jobs/{job_id}",
        response_list=[{"json": response} for response in responses],
    )
    return mocker, len(responses)


def _generate_project(
    name="project-name",
    description="project description",
    labels=None,
    annotations=None,
    created=None,
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
            created=created or datetime.datetime.utcnow(),
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
    job_id: typing.Optional[str] = None,
    operational_status: typing.Optional[mlrun.api.schemas.ProjectState] = None,
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
    body["attributes"]["operational_status"] = (
        operational_status.value
        if operational_status
        else body["attributes"]["admin_status"]
    )
    if job_id:
        body["relationships"] = {
            "last_job": {"data": {"id": job_id}},
        }
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
    mlrun_project_dict = json.loads(request_body["data"]["attributes"]["mlrun_project"])
    expected_project_dict = project.dict(
        exclude_unset=True,
        exclude={
            "metadata": {"name", "created", "labels", "annotations"},
            "spec": {"description", "desired_state"},
            "status": {"state"},
        },
    )
    for field in ["metadata", "spec", "status"]:
        assert mlrun_project_dict[field] is not None
        expected_project_dict.setdefault(field, {})
    assert request_body["data"]["attributes"]["mlrun_project"] == json.dumps(
        expected_project_dict
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
