# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import asyncio
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
from aioresponses import CallbackResult
from requests.cookies import cookiejar_from_dict

import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.config
import mlrun.errors
from tests.common_fixtures import aioresponses_mock


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://iguazio-api-url:8080"
    mlrun.config.config._iguazio_api_url = api_url
    return api_url


@pytest.fixture()
async def iguazio_client(
    api_url: str,
    request,
) -> mlrun.api.utils.clients.iguazio.Client:
    if request.param == "async":
        client = mlrun.api.utils.clients.iguazio.AsyncClient()
    else:
        client = mlrun.api.utils.clients.iguazio.Client()

    # force running init again so the configured api url will be used
    client.__init__()
    client._wait_for_job_completion_retry_interval = 0
    client._wait_for_project_terminal_state_retry_interval = 0

    # inject the request param into client, so we can use it in tests
    setattr(client, "mode", request.param)
    return client


def patch_restful_request(
    is_client_sync: bool,
    requests_mock: requests_mock_package.Mocker,
    aioresponses_mock: aioresponses_mock,
    method: str,
    url: str,
    callback: typing.Optional[typing.Callable] = None,
    status_code: typing.Optional[int] = None,
):
    """
    Consolidating the requests_mock / aioresponses library to mock a RESTful request.
    """
    kwargs = {}
    if is_client_sync:
        if callback:
            kwargs["json"] = callback
        if status_code:
            kwargs["status_code"] = status_code
        requests_mock.request(
            method,
            url,
            **kwargs,
        )
    else:
        if callback:
            kwargs["callback"] = callback
        if status_code:
            kwargs["status"] = status_code
        aioresponses_mock.add(
            url,
            method,
            **kwargs,
        )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_verify_request_session_success(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
    aioresponses_mock: aioresponses_mock,
):
    mock_request_headers = starlette.datastructures.Headers(
        {"cookie": "session=some-session-cookie"}
    )
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = mock_request_headers

    mock_response_headers = _generate_session_verification_response_headers()

    def _verify_session_mock(*args, **kwargs):
        response = {}
        if iguazio_client.is_sync:
            request, context = args
            request_headers = request.headers
            context.headers = mock_response_headers
        else:
            request_headers = kwargs["headers"]
        for header_key, header_value in mock_request_headers.items():
            assert request_headers[header_key] == header_value
        if iguazio_client.is_sync:
            return response
        else:
            return CallbackResult(headers=mock_response_headers)

    def _verify_session_with_body_mock(*args, **kwargs):
        response = {
            "data": {
                "attributes": {
                    "username": "some-user",
                    "context": {
                        "authentication": {
                            "user_id": "some-user-id",
                            "tenant_id": "some-tenant-id",
                            "group_ids": [
                                "some-group-id-1,some-group-id-2",
                            ],
                            "mode": "normal",
                        },
                    },
                },
            },
        }
        if iguazio_client.is_sync:
            request, context = args
            request_headers = request.headers
            context.headers = mock_response_headers
        else:
            request_headers = kwargs["headers"]
        for header_key, header_value in mock_request_headers.items():
            assert request_headers[header_key] == header_value

        if iguazio_client.is_sync:
            return response
        else:
            return CallbackResult(payload=response, headers=mock_response_headers)

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"
    for test_case in [
        {
            "response_json": _verify_session_mock,
        },
        {
            "response_json": _verify_session_with_body_mock,
        },
    ]:
        patch_restful_request(
            iguazio_client.is_sync,
            requests_mock,
            aioresponses_mock,
            method="POST",
            url=url,
            callback=test_case["response_json"],
        )

        auth_info = await maybe_coroutine(
            iguazio_client.verify_request_session(mock_request)
        )
        _assert_auth_info_from_session_verification_mock_response_headers(
            auth_info, mock_response_headers
        )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_verify_request_session_failure(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
    aioresponses_mock: aioresponses_mock,
):
    mock_request = fastapi.Request({"type": "http"})
    mock_request._headers = starlette.datastructures.Headers()
    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"
    patch_restful_request(
        iguazio_client.is_sync,
        requests_mock,
        aioresponses_mock,
        method="POST",
        url=url,
        status_code=http.HTTPStatus.UNAUTHORIZED.value,
    )
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        await maybe_coroutine(iguazio_client.verify_request_session(mock_request))
        assert exc.value.status_code == http.HTTPStatus.UNAUTHORIZED.value


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_verify_session_success(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
    aioresponses_mock: aioresponses_mock,
):
    session = "some-session"
    mock_response_headers = _generate_session_verification_response_headers()

    def _verify_session_mock(*args, **kwargs):
        if iguazio_client.is_sync:
            request, context = args
            _verify_request_cookie(request.headers, session)
            context.headers = mock_response_headers
        else:
            _verify_request_cookie(kwargs, session)

        return (
            {}
            if iguazio_client.is_sync
            else CallbackResult(headers=mock_response_headers)
        )

    url = f"{api_url}/api/{mlrun.mlconf.httpdb.authentication.iguazio.session_verification_endpoint}"
    patch_restful_request(
        iguazio_client.is_sync,
        requests_mock,
        aioresponses_mock,
        method="POST",
        url=url,
        callback=_verify_session_mock,
    )
    auth_info = await maybe_coroutine(iguazio_client.verify_session(session))
    _assert_auth_info_from_session_verification_mock_response_headers(
        auth_info, mock_response_headers
    )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_get_grafana_service_url_success(
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
    grafana_url = await maybe_coroutine(
        iguazio_client.try_get_grafana_service_url("session-cookie")
    )
    assert grafana_url == expected_grafana_url


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_get_grafana_service_url_ignoring_disabled_service(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    grafana_service = {"spec": {"kind": "grafana"}, "status": {"state": "disabled"}}
    response_body = _generate_app_services_manifests_body([grafana_service])
    requests_mock.get(f"{api_url}/api/app_services_manifests", json=response_body)
    grafana_url = await maybe_coroutine(
        iguazio_client.try_get_grafana_service_url("session-cookie")
    )
    assert grafana_url is None


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_get_grafana_service_url_no_grafana_exists(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    response_body = _generate_app_services_manifests_body([])
    requests_mock.get(f"{api_url}/api/app_services_manifests", json=response_body)
    grafana_url = await maybe_coroutine(
        iguazio_client.try_get_grafana_service_url("session-cookie")
    )
    assert grafana_url is None


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_get_grafana_service_url_no_urls(
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
    grafana_url = await maybe_coroutine(
        iguazio_client.try_get_grafana_service_url("session-cookie")
    )
    assert grafana_url is None


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_get_or_create_access_key_success(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    planes = [
        mlrun.api.utils.clients.iguazio.SessionPlanes.control,
    ]
    access_key_id = "some-id"
    session = "1234"

    def _get_or_create_access_key_mock(status_code, request, context):
        _verify_request_cookie(request.headers, session)
        context.status_code = status_code
        expected_request_body = {
            "data": {
                "type": "access_key",
                "attributes": {"label": "MLRun", "planes": planes},
            }
        }
        assert (
            deepdiff.DeepDiff(
                expected_request_body,
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        return {"data": {"id": access_key_id}}

    # mock creation
    requests_mock.post(
        f"{api_url}/api/self/get_or_create_access_key",
        json=functools.partial(
            _get_or_create_access_key_mock, http.HTTPStatus.CREATED.value
        ),
    )
    returned_access_key = await maybe_coroutine(
        iguazio_client.get_or_create_access_key(session, planes)
    )
    assert access_key_id == returned_access_key

    # mock get
    requests_mock.post(
        f"{api_url}/api/self/get_or_create_access_key",
        json=functools.partial(
            _get_or_create_access_key_mock, http.HTTPStatus.OK.value
        ),
    )
    returned_access_key = await maybe_coroutine(
        iguazio_client.get_or_create_access_key(session, planes)
    )
    assert access_key_id == returned_access_key


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_get_project_owner(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    owner_username = "some-username"
    project = _generate_project(owner=owner_username)
    session = "1234"
    owner_access_key = "some-access-key"

    def verify_get(request, context):
        assert request.qs == {
            "include": ["owner"],
            "enrich_owner_access_key": ["true"],
        }
        context.status_code = http.HTTPStatus.OK.value
        _verify_project_request_headers(request.headers, session)
        return {
            "data": _build_project_response(
                iguazio_client, project, owner_access_key=owner_access_key
            )
        }

    # mock project response so store will update
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json=verify_get,
    )
    project_owner = await maybe_coroutine(
        iguazio_client.get_project_owner(
            session,
            project.metadata.name,
        )
    )
    assert project_owner.username == owner_username
    assert project_owner.access_key == owner_access_key


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_list_project_with_updated_after(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session = "1234"
    updated_after = datetime.datetime.now(tz=datetime.timezone.utc)

    def verify_list(request, context):
        assert request.qs == {
            "filter[updated_at]": [
                f"[$gt]{updated_after.isoformat().split('+')[0]}Z".lower()
            ],
            "include": ["owner"],
            "page[size]": [
                str(
                    mlrun.mlconf.httpdb.projects.iguazio_list_projects_default_page_size
                )
            ],
        }
        context.status_code = http.HTTPStatus.OK.value
        _verify_project_request_headers(request.headers, session)
        return {"data": [_build_project_response(iguazio_client, project)]}

    # mock project response so store will update
    requests_mock.get(
        f"{api_url}/api/projects",
        json=verify_list,
    )
    await maybe_coroutine(
        iguazio_client.list_projects(
            session,
            updated_after,
        )
    )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_list_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    mock_projects = [
        {"name": "project-name-1"},
        {"name": "project-name-2", "description": "project-description-2"},
        {"name": "project-name-3", "owner": "some-owner"},
        {"name": "project-name-4", "labels": {"key": "value"}},
        {
            "name": "project-name-5",
            "annotations": {"annotation-key": "annotation-value"},
        },
        {
            "name": "project-name-6",
            "description": "project-description-4",
            "owner": "some-owner",
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
                    owner=mock_project.get("owner", None),
                ),
            )
            for mock_project in mock_projects
        ]
    }
    requests_mock.get(f"{api_url}/api/projects", json=response_body)
    projects, latest_updated_at = await maybe_coroutine(
        iguazio_client.list_projects(None)
    )
    for index, project in enumerate(projects):
        assert project.metadata.name == mock_projects[index]["name"]
        assert project.spec.description == mock_projects[index].get("description")
        assert project.spec.owner == mock_projects[index].get("owner")
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


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_create_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    await _create_project_and_assert(api_url, iguazio_client, requests_mock, project)


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_create_project_failures(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    """
    The exception handling is generic so no need to test it for every action (read/update/delete).
    There are basically 2 options:
    1. Validations failure - in this case job won't be triggered, and we'll get an error (http) response from iguazio
    2. Processing failure - this will happen inside the job, so we'll see the job finishing with failed state, sometimes
    the job result will have nice error messages for us
    """
    session = "1234"
    project = _generate_project()

    # mock validation failure
    error_message = "project name invalid or something"
    requests_mock.post(
        f"{api_url}/api/projects",
        status_code=http.HTTPStatus.BAD_REQUEST.value,
        json={
            "errors": [
                {"status": http.HTTPStatus.BAD_REQUEST.value, "detail": error_message}
            ]
        },
    )

    with pytest.raises(
        mlrun.errors.MLRunBadRequestError, match=rf"(.*){error_message}(.*)"
    ):
        await maybe_coroutine(
            iguazio_client.create_project(
                session,
                project,
            )
        )

    # mock job failure - with nice error message in result
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session, job_id
        ),
    )
    error_message = "failed creating project in Nuclio for example"
    job_result = json.dumps(
        {"status": http.HTTPStatus.BAD_REQUEST.value, "message": error_message}
    )
    _mock_job_progress(
        api_url,
        requests_mock,
        session,
        job_id,
        mlrun.api.utils.clients.iguazio.JobStates.failed,
        job_result,
    )

    with pytest.raises(
        mlrun.errors.MLRunBadRequestError, match=rf"(.*){error_message}(.*)"
    ):
        await maybe_coroutine(
            iguazio_client.create_project(
                session,
                project,
            )
        )

    # mock job failure - without nice error message (shouldn't happen, but let's test)
    _mock_job_progress(
        api_url,
        requests_mock,
        session,
        job_id,
        mlrun.api.utils.clients.iguazio.JobStates.failed,
    )

    with pytest.raises(mlrun.errors.MLRunRuntimeError):
        await maybe_coroutine(
            iguazio_client.create_project(
                session,
                project,
            )
        )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_create_project_minimal_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(
            name="some-name",
        ),
    )
    await _create_project_and_assert(api_url, iguazio_client, requests_mock, project)


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_create_project_without_wait(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session = "1234"
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session, job_id
        ),
    )
    is_running_in_background = await maybe_coroutine(
        iguazio_client.create_project(session, project, wait_for_completion=False)
    )
    assert is_running_in_background is True


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_update_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project()
    session = "1234"

    def verify_store_update(request, context):
        _assert_project_creation(iguazio_client, request.json(), project)
        context.status_code = http.HTTPStatus.OK.value
        _verify_project_request_headers(request.headers, session)
        return {"data": _build_project_response(iguazio_client, project)}

    requests_mock.put(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json=verify_store_update,
    )
    await maybe_coroutine(
        iguazio_client.update_project(
            session,
            project.metadata.name,
            project,
        )
    )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_update_project_remove_labels_and_annotations(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project = _generate_project(name="empty-labels", labels={}, annotations={})
    project_without_labels = _generate_project(name="no-labels")
    project_without_labels.metadata.labels = None
    project_without_labels.metadata.annotations = None
    session = "1234"

    def verify_empty_labels_and_annotations(request, context):
        request_body = request.json()
        assert request_body["data"]["attributes"]["labels"] == []
        assert request_body["data"]["attributes"]["annotations"] == []

        context.status_code = http.HTTPStatus.OK.value
        return {"data": _build_project_response(iguazio_client, project)}

    def verify_no_labels_and_annotations_in_request(request, context):
        request_body = request.json()
        assert "labels" not in request_body["data"]["attributes"]
        assert "annotations" not in request_body["data"]["attributes"]

        context.status_code = http.HTTPStatus.OK.value
        return {"data": _build_project_response(iguazio_client, project)}

    requests_mock.put(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json=verify_empty_labels_and_annotations,
    )
    requests_mock.put(
        f"{api_url}/api/projects/__name__/{project_without_labels.metadata.name}",
        json=verify_no_labels_and_annotations_in_request,
    )

    await maybe_coroutine(
        iguazio_client.update_project(
            session,
            project.metadata.name,
            project,
        )
    )
    await maybe_coroutine(
        iguazio_client.update_project(
            session,
            project_without_labels.metadata.name,
            project_without_labels,
        )
    )


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_delete_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    job_id = "928145d5-4037-40b0-98b6-19a76626d797"
    session = "1234"

    requests_mock.delete(
        f"{api_url}/api/projects",
        json=functools.partial(_verify_deletion, project_name, session, job_id),
    )
    mocker, num_of_calls_until_completion = _mock_job_progress(
        api_url, requests_mock, session, job_id
    )
    is_running_in_background = await maybe_coroutine(
        iguazio_client.delete_project(session, project_name)
    )
    assert is_running_in_background is False
    assert mocker.call_count == num_of_calls_until_completion

    # assert ignoring (and not exploding) on not found
    requests_mock.delete(
        f"{api_url}/api/projects", status_code=http.HTTPStatus.NOT_FOUND.value
    )
    await maybe_coroutine(iguazio_client.delete_project(session, project_name))

    # TODO: not sure really needed
    # assert correctly propagating 412 errors (will be returned when project has resources)
    requests_mock.delete(
        f"{api_url}/api/projects", status_code=http.HTTPStatus.PRECONDITION_FAILED.value
    )
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        await maybe_coroutine(iguazio_client.delete_project(session, project_name))


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_delete_project_without_wait(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    job_id = "928145d5-4037-40b0-98b6-19a76626d797"
    session = "1234"

    requests_mock.delete(
        f"{api_url}/api/projects",
        json=functools.partial(_verify_deletion, project_name, session, job_id),
    )
    is_running_in_background = await maybe_coroutine(
        iguazio_client.delete_project(session, project_name, wait_for_completion=False)
    )
    assert is_running_in_background is True


@pytest.mark.parametrize("iguazio_client", ("async", "sync"), indirect=True)
@pytest.mark.asyncio
async def test_format_as_leader_project(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
):
    project = _generate_project()
    iguazio_project = await maybe_coroutine(
        iguazio_client.format_as_leader_project(project)
    )
    assert (
        deepdiff.DeepDiff(
            _build_project_response(iguazio_client, project),
            iguazio_project.data,
            ignore_order=True,
            exclude_paths=[
                "root['attributes']['updated_at']",
                "root['attributes']['operational_status']",
            ],
        )
        == {}
    )


def _generate_session_verification_response_headers(
    username="some-user",
    session="some-access-key",
    user_id="some-user-id",
    user_group_ids="some-group-id-1,some-group-id-2",
    planes="control,data",
):
    return {
        "X-Remote-User": username,
        "X-V3io-Session-Key": session,
        "x-user-id": user_id,
        "x-user-group-ids": user_group_ids,
        "x-v3io-session-planes": planes,
    }


def _assert_auth_info_from_session_verification_mock_response_headers(
    auth_info: mlrun.api.schemas.AuthInfo, response_headers: dict
):
    _assert_auth_info(
        auth_info,
        response_headers["X-Remote-User"],
        response_headers["X-V3io-Session-Key"],
        response_headers["X-V3io-Session-Key"],
        response_headers["x-user-id"],
        response_headers["x-user-group-ids"].split(","),
    )


def _assert_auth_info(
    auth_info: mlrun.api.schemas.AuthInfo,
    username: str,
    session: str,
    data_session: str,
    user_id: str,
    user_group_ids: typing.List[str],
):
    assert auth_info.username == username
    assert auth_info.session == session
    assert auth_info.user_id == user_id
    assert auth_info.user_group_ids == user_group_ids
    # we returned data in planes so a data session as well
    assert auth_info.data_session == data_session


async def _create_project_and_assert(
    api_url: str,
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    requests_mock: requests_mock_package.Mocker,
    project: mlrun.api.schemas.Project,
):
    session = "1234"
    job_id = "1d4c9d25-9c5c-4a34-b052-c1d3665fec5e"

    requests_mock.post(
        f"{api_url}/api/projects",
        json=functools.partial(
            _verify_creation, iguazio_client, project, session, job_id
        ),
    )
    mocker, num_of_calls_until_completion = _mock_job_progress(
        api_url, requests_mock, session, job_id
    )
    requests_mock.get(
        f"{api_url}/api/projects/__name__/{project.metadata.name}",
        json={"data": _build_project_response(iguazio_client, project)},
    )
    is_running_in_background = await maybe_coroutine(
        iguazio_client.create_project(
            session,
            project,
        )
    )
    assert is_running_in_background is False
    assert mocker.call_count == num_of_calls_until_completion


def _verify_deletion(project_name, session, job_id, request, context):
    assert request.json()["data"]["attributes"]["name"] == project_name
    assert (
        request.headers["igz-project-deletion-strategy"]
        == mlrun.api.schemas.DeletionStrategy.default().to_iguazio_deletion_strategy()
    )
    _verify_project_request_headers(request.headers, session)
    context.status_code = http.HTTPStatus.ACCEPTED.value
    return {"data": {"type": "job", "id": job_id}}


def _verify_creation(iguazio_client, project, session, job_id, request, context):
    _assert_project_creation(iguazio_client, request.json(), project)
    context.status_code = http.HTTPStatus.CREATED.value
    _verify_project_request_headers(request.headers, session)
    return {
        "data": _build_project_response(
            iguazio_client, project, job_id, mlrun.api.schemas.ProjectState.creating
        )
    }


def _verify_request_cookie(headers: dict, session: str):
    expected_session_value = f'session=j:{{"sid": "{session}"}}'
    if "Cookie" in headers:
        assert headers["Cookie"] == expected_session_value
    elif "cookies" in headers:

        # in async client we get the `cookies` key while it contains the cookies in form of a dict
        # use requests to construct it back to a string as expected above
        cookie = "; ".join(
            list(
                map(
                    lambda x: f"{x[0]}={x[1]}",
                    cookiejar_from_dict(headers["cookies"]).items(),
                )
            )
        )
        assert cookie == expected_session_value
    else:
        raise AssertionError("No cookie found in headers")


def _verify_project_request_headers(headers: dict, session: str):
    _verify_request_cookie(headers, session)
    assert headers[mlrun.api.schemas.HeaderNames.projects_role] == "mlrun"


def _mock_job_progress(
    api_url,
    requests_mock,
    session: str,
    job_id: str,
    terminal_job_state: str = mlrun.api.utils.clients.iguazio.JobStates.completed,
    job_result: str = "",
):
    def _mock_get_job(state, result, session, request, context):
        context.status_code = http.HTTPStatus.OK.value
        assert request.headers["Cookie"] == f'session=j:{{"sid": "{session}"}}'
        return {"data": {"attributes": {"state": state, "result": result}}}

    responses = [
        functools.partial(
            _mock_get_job,
            mlrun.api.utils.clients.iguazio.JobStates.in_progress,
            job_result,
            session,
        ),
        functools.partial(
            _mock_get_job,
            mlrun.api.utils.clients.iguazio.JobStates.in_progress,
            job_result,
            session,
        ),
        functools.partial(_mock_get_job, terminal_job_state, job_result, session),
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
    owner="project-owner",
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
            owner=owner,
            some_extra_field="some value",
        ),
        status=mlrun.api.schemas.ProjectStatus(
            some_extra_field="some value",
        ),
    )


def _build_project_response(
    iguazio_client: mlrun.api.utils.clients.iguazio.Client,
    project: mlrun.api.schemas.Project,
    job_id: typing.Optional[str] = None,
    operational_status: typing.Optional[mlrun.api.schemas.ProjectState] = None,
    owner_access_key: typing.Optional[str] = None,
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
    if project.spec.owner:
        body["attributes"]["owner_username"] = project.spec.owner
    if owner_access_key:
        body["attributes"]["owner_access_key"] = owner_access_key
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
            "spec": {"description", "desired_state", "owner"},
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


async def maybe_coroutine(o):
    if asyncio.iscoroutine(o):
        return await o
    return o
