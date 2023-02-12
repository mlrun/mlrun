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
import http

import fastapi.testclient
import httpx
import pytest
import sqlalchemy.orm

import mlrun
import mlrun.api.main
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import mlrun.api.utils.singletons.scheduler
import tests.api.api.utils
from mlrun.api import schemas
from mlrun.api.utils.singletons.db import get_db
from tests.common_fixtures import aioresponses_mock

ORIGINAL_VERSIONED_API_PREFIX = mlrun.api.main.BASE_VERSIONED_API_PREFIX


async def do_nothing():
    pass


def test_list_schedules(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    resp = client.get("projects/default/schedules")
    assert resp.status_code == http.HTTPStatus.OK.value, "status"
    assert "schedules" in resp.json(), "no schedules"

    labels_1 = {
        "label1": "value1",
    }
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = mlrun.mlconf.default_project
    get_db().create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
        mlrun.mlconf.httpdb.scheduling.default_concurrency_limit,
        labels_1,
    )

    labels_2 = {
        "label2": "value2",
    }
    schedule_name_2 = "schedule-name-2"
    get_db().create_schedule(
        db,
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
        mlrun.mlconf.httpdb.scheduling.default_concurrency_limit,
        labels_2,
    )

    _get_and_assert_single_schedule(client, {"labels": "label1"}, schedule_name)
    _get_and_assert_single_schedule(client, {"labels": "label2"}, schedule_name_2)
    _get_and_assert_single_schedule(client, {"labels": "label1=value1"}, schedule_name)
    _get_and_assert_single_schedule(
        client, {"labels": "label2=value2"}, schedule_name_2
    )


@pytest.mark.parametrize(
    "method, body, expected_status, expected_body, expected_response_from_chief, create_project",
    [
        # project doesn't exist, expecting to fail before forwarding to chief
        [
            "POST",
            tests.api.api.utils.compile_schedule(),
            http.HTTPStatus.NOT_FOUND.value,
            {"detail": "MLRunNotFoundError('Project {project_name} does not exist')"},
            False,
            False,
        ],
        # project exists, expecting to create
        [
            "POST",
            tests.api.api.utils.compile_schedule(),
            http.HTTPStatus.CREATED.value,
            {},
            True,
            True,
        ],
    ],
)
@pytest.mark.asyncio
async def test_redirection_from_worker_to_chief_create_schedule(
    db: sqlalchemy.orm.Session,
    async_client: httpx.AsyncClient,
    aioresponses_mock: aioresponses_mock,
    method: str,
    body: dict,
    expected_status: int,
    expected_body: dict,
    expected_response_from_chief: bool,
    create_project: bool,
):
    project = "test-project"
    endpoint, chief_mocked_url = _prepare_test_redirection_from_worker_to_chief(
        project=project,
    )
    _format_expected_body(expected_body, project_name=project)

    if create_project:
        await tests.api.api.utils.create_project_async(async_client, project)
    if expected_response_from_chief:
        aioresponses_mock.post(
            chief_mocked_url, status=expected_status, payload=expected_body
        )
    response = await async_client.post(endpoint, data=body)
    assert response.status_code == expected_status
    assert response.json() == expected_body


@pytest.mark.parametrize(
    "method, body, expected_status, expected_body",
    [
        # deleting schedule failed for unknown reason
        [
            "DELETE",
            None,
            http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            {"detail": "Unknown error"},
        ],
        # deleting schedule succeeded
        [
            "DELETE",
            None,
            http.HTTPStatus.NOT_FOUND.value,
            {},
        ],
        # we don't check if the project exists in update schedule, but rather query from the db and raise exception
        # if schedule doesn't exist
        [
            "PUT",
            tests.api.api.utils.compile_schedule(),
            http.HTTPStatus.NOT_FOUND.value,
            {
                "detail": "MLRunNotFoundError('Schedule not found: project={project_name}, name={schedule_name}')"
            },
        ],
        # updating schedule failed for unknown reason
        [
            "PUT",
            tests.api.api.utils.compile_schedule(),
            http.HTTPStatus.NOT_FOUND.value,
            {
                "detail": "MLRunNotFoundError('Schedule not found: project={project_name}, name={schedule_name}')"
            },
        ],
        # project exists, expecting to create
        ["PUT", tests.api.api.utils.compile_schedule(), http.HTTPStatus.OK.value, {}],
    ],
)
@pytest.mark.asyncio
async def test_redirection_from_worker_to_chief_schedule(
    db: sqlalchemy.orm.Session,
    async_client: httpx.AsyncClient,
    aioresponses_mock: aioresponses_mock,
    method: str,
    body: dict,
    expected_status: int,
    expected_body: dict,
):
    project_name = "test-project"
    schedule_name = "test_schedule"
    endpoint, chief_mocked_url = _prepare_test_redirection_from_worker_to_chief(
        project=project_name, endpoint_suffix=schedule_name
    )

    # template the expected body
    _format_expected_body(
        expected_body, project_name=project_name, schedule_name=schedule_name
    )

    # what the chief will return
    aioresponses_mock.add(
        chief_mocked_url,
        method,
        status=expected_status,
        payload=expected_body,
    )
    response = await async_client.request(method, endpoint, data=body)
    assert response.status_code == expected_status
    assert response.json() == expected_body
    aioresponses_mock.assert_called_once()


@pytest.mark.parametrize(
    "expected_status, expected_body",
    [
        [
            # invoking schedule failed for unknown reason
            http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            {"detail": {"reason": "Unknown error"}},
        ],
        [
            # expecting to succeed
            http.HTTPStatus.NOT_FOUND.value,
            {},
        ],
    ],
)
@pytest.mark.asyncio
async def test_redirection_from_worker_to_chief_delete_schedules(
    db: sqlalchemy.orm.Session,
    async_client: httpx.AsyncClient,
    aioresponses_mock: aioresponses_mock,
    expected_status: int,
    expected_body: dict,
):

    # so get_scheduler().list_schedules, which is called in the delete_schedules endpoint, will return something
    await mlrun.api.utils.singletons.scheduler.initialize_scheduler()
    endpoint, chief_mocked_url = _prepare_test_redirection_from_worker_to_chief(
        project="test-project",
    )

    aioresponses_mock.delete(
        chief_mocked_url,
        status=expected_status,
        payload=expected_body,
    )

    response = await async_client.delete(endpoint)
    assert response.status_code == expected_status
    assert response.json() == expected_body


@pytest.mark.parametrize(
    "expected_status, expected_body",
    [
        [
            # invoking schedule failed for unknown reason
            http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            {"detail": {"reason": "unknown error"}},
        ],
        [
            # expecting to succeed
            http.HTTPStatus.OK.value,
            {},
        ],
    ],
)
@pytest.mark.asyncio
async def test_redirection_from_worker_to_chief_invoke_schedule(
    db: sqlalchemy.orm.Session,
    async_client: httpx.AsyncClient,
    aioresponses_mock: aioresponses_mock,
    expected_status: int,
    expected_body: dict,
):
    endpoint, chief_mocked_url = _prepare_test_redirection_from_worker_to_chief(
        project="test-project", endpoint_suffix="test_scheduler/invoke"
    )

    aioresponses_mock.post(
        chief_mocked_url,
        status=expected_status,
        payload=expected_body,
    )

    response = await async_client.post(endpoint)
    assert response.status_code == expected_status
    assert response.json() == expected_body


def _prepare_test_redirection_from_worker_to_chief(project, endpoint_suffix=""):
    mlrun.mlconf.httpdb.clusterization.chief.url = "http://chief:8080"
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    mlrun.mlconf.httpdb.clusterization.worker.request_timeout = 3
    endpoint = f"projects/{project}/schedules"
    if endpoint_suffix:
        endpoint = f"{endpoint}/{endpoint_suffix}"
    chief_mocked_url = f"{mlrun.mlconf.httpdb.clusterization.chief.url}{ORIGINAL_VERSIONED_API_PREFIX}/{endpoint}"
    return endpoint, chief_mocked_url


def _get_and_assert_single_schedule(
    client: fastapi.testclient.TestClient, get_params: dict, schedule_name: str
):
    resp = client.get("projects/default/schedules", params=get_params)
    assert resp.status_code == http.HTTPStatus.OK.value, "status"
    result = resp.json()["schedules"]
    assert len(result) == 1
    assert result[0]["name"] == schedule_name


def _format_expected_body(expected_body: dict, **kwargs):
    if "detail" in expected_body:
        expected_body["detail"] = expected_body["detail"].format(**kwargs)
    return expected_body
