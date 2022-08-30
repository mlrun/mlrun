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
import uuid

import fastapi.testclient
import sqlalchemy.orm

import mlrun
import mlrun.api.main
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
import tests.api.api.utils
from mlrun.api import schemas
from mlrun.api.utils.singletons.db import get_db

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


def test_redirection_from_worker_to_chief_create_schedule(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project = "test-project"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/schedules"

    for test_case in [
        # project doesn't exists, expecting to fail before forwarding to chief
        {
            "body": _create_schedule(),
            "expected_status": http.HTTPStatus.NOT_FOUND.value,
            "expected_body": {
                "detail": {
                    "reason": f"MLRunNotFoundError('Project {project} does not exist')"
                }
            },
            "expect_response_from_chief": False,
            "create_project": False,
        },
        # project exists, expecting to create
        {
            "body": _create_schedule(),
            "expected_status": http.HTTPStatus.CREATED.value,
            "expected_body": {},
        },
    ]:
        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")
        body = test_case.get("body")
        create_project = test_case.get("create_project", True)
        expect_response_from_chief = test_case.get("expect_response_from_chief", True)

        if create_project:
            tests.api.api.utils.create_project(client, project)
        if expect_response_from_chief:
            httpserver.expect_ordered_request(
                endpoint, method="POST"
            ).respond_with_json(expected_response, status=expected_status)
            url = httpserver.url_for("")
            mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.post(endpoint, data=body)
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_redirection_from_worker_to_chief_update_schedule(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project = "test-project"
    schedule_name = "test_scheduler"
    endpoint = (
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/schedules/{schedule_name}"
    )

    for test_case in [
        # we don't check if the project exists in update schedule, but rather query from the db and raise exception
        # if schedule doesn't exists
        {
            "body": _create_schedule(),
            "expected_status": http.HTTPStatus.NOT_FOUND.value,
            "expected_body": {
                "detail": {
                    "reason": f"MLRunNotFoundError('Schedule not found: project={project}, name={schedule_name}')"
                }
            },
        },
        # updating schedule failed for unknown reason
        {
            "body": _create_schedule(),
            "expected_status": http.HTTPStatus.NOT_FOUND.value,
            "expected_body": {
                "detail": {
                    "reason": f"MLRunNotFoundError('Schedule not found: project={project}, name={schedule_name}')"
                }
            },
        },
        # project exists, expecting to create
        {
            "body": _create_schedule(),
            "expected_status": http.HTTPStatus.OK.value,
            "expected_body": {},
        },
    ]:
        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")
        body = test_case.get("body")

        httpserver.expect_ordered_request(endpoint, method="PUT").respond_with_json(
            expected_response, status=expected_status
        )
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.put(endpoint, data=body)
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_redirection_from_worker_to_chief_delete_schedule(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project = "test-project"
    schedule_name = "test_scheduler"
    endpoint = (
        f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/schedules/{schedule_name}"
    )

    for test_case in [
        # deleting schedule failed for unknown reason
        {
            "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "expected_body": {"detail": {"reason": "Unknown error"}},
        },
        # deleting schedule succeeded
        {
            "expected_status": http.HTTPStatus.NOT_FOUND.value,
            "expected_body": {},
        },
    ]:
        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")

        httpserver.expect_ordered_request(endpoint, method="DELETE").respond_with_json(
            expected_response, status=expected_status
        )
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.delete(endpoint)
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_redirection_from_worker_to_chief_delete_schedules(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project = "test-project"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/schedules"

    for test_case in [
        # deleting schedule failed for unknown reason
        {
            "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "expected_body": {"detail": {"reason": "Unknown error"}},
        },
        # deleting project's schedules succeeded
        {
            "expected_status": http.HTTPStatus.NOT_FOUND.value,
            "expected_body": {},
        },
    ]:
        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")

        httpserver.expect_ordered_request(endpoint, method="DELETE").respond_with_json(
            expected_response, status=expected_status
        )
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.delete(endpoint)
        assert response.status_code == expected_status
        assert response.json() == expected_response


def test_redirection_from_worker_to_chief_invoke_schedule(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project = "test-project"
    schedule_name = "test_scheduler"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}/schedules/{schedule_name}/invoke"

    for test_case in [
        # invoking schedule failed for unknown reason
        {
            "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "expected_body": {"detail": {"reason": "unknown error"}},
        },
        # expecting to succeed
        {
            "expected_status": http.HTTPStatus.OK.value,
            "expected_body": {},
        },
    ]:
        expected_status = test_case.get("expected_status")
        expected_response = test_case.get("expected_body")

        httpserver.expect_ordered_request(endpoint, method="POST").respond_with_json(
            expected_response, status=expected_status
        )
        url = httpserver.url_for("")
        mlrun.mlconf.httpdb.clusterization.chief.url = url
        response = client.post(endpoint)
        assert response.status_code == expected_status
        assert response.json() == expected_response


def _get_and_assert_single_schedule(
    client: fastapi.testclient.TestClient, get_params: dict, schedule_name: str
):
    resp = client.get("projects/default/schedules", params=get_params)
    assert resp.status_code == http.HTTPStatus.OK.value, "status"
    result = resp.json()["schedules"]
    assert len(result) == 1
    assert result[0]["name"] == schedule_name


def _create_schedule(schedule_name: str = None, to_json: bool = True):
    if not schedule_name:
        schedule_name = f"schedule-name-{str(uuid.uuid4())}"
    schedule = mlrun.api.schemas.ScheduleInput(
        name=schedule_name,
        kind=mlrun.api.schemas.ScheduleKinds.job,
        scheduled_object={"metadata": {"name": "something"}},
        cron_trigger=mlrun.api.schemas.ScheduleCronTrigger(year=1999),
    )
    if not to_json:
        return schedule
    return mlrun.utils.dict_to_json(schedule.dict())
