# Copyright 2023 Iguazio
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

import unittest.mock
import uuid
from http import HTTPStatus

import httpx
from fastapi.testclient import TestClient

import mlrun.common.schemas

import framework.utils.clients.iguazio.v3

PROJECT = "project-name"


def setup_iguazio_v3_async_client_mock(
    monkeypatch, username="username", session="session", data_session="data_session"
):
    """
    Set up a properly configured mock for Iguazio v3 AsyncClient

    The mock's verify_request_session method returns a proper AuthInfo object instead of
    a mock, which prevents pickling errors when the project object is serialized.

    Args:
        monkeypatch: pytest monkeypatch fixture
        username: Username to return in AuthInfo (default: "username")
        session: Session to return in AuthInfo (default: "session")
        data_session: Data session to return in AuthInfo (default: "data_session")

    Returns:
        The configured mock client
    """
    mock_client = unittest.mock.AsyncMock()
    mock_client.verify_request_session = unittest.mock.AsyncMock(
        return_value=mlrun.common.schemas.auth.AuthInfo(
            username=username,
            session=session,
            data_session=data_session,
        )
    )
    monkeypatch.setattr(
        framework.utils.clients.iguazio.v3,
        "AsyncClient",
        lambda *args, **kwargs: mock_client,
    )
    return mock_client


def create_project(
    client: TestClient,
    project_name: str = PROJECT,
    artifact_path=None,
    source="source",
    load_source_on_run=False,
    endpoint_prefix="",
    default_function_node_selector=None,
):
    project = _create_project_obj(
        project_name,
        artifact_path,
        source,
        load_source_on_run,
        default_function_node_selector,
    )
    resp = client.post(f"{endpoint_prefix}projects", json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def compile_schedule(schedule_name: str | None = None, to_json: bool = True):
    if not schedule_name:
        schedule_name = f"schedule-name-{str(uuid.uuid4())}"
    schedule = mlrun.common.schemas.ScheduleInput(
        name=schedule_name,
        kind=mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object={"metadata": {"name": "something"}},
        cron_trigger=mlrun.common.schemas.ScheduleCronTrigger(year=1999),
    )
    if not to_json:
        return schedule
    return mlrun.utils.dict_to_json(schedule.dict())


async def create_project_async(
    async_client: httpx.AsyncClient, project_name: str = PROJECT
):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = await async_client.post(
        "projects",
        json=project.dict(),
    )
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def assert_pagination_info(
    response,
    expected_page: int,
    expected_results_count: int,
    expected_page_size: int,
    expected_first_result_name: str,
    entity_name: str,
    entity_identifier_name: str,
):
    assert response.status_code == HTTPStatus.OK.value, (
        f"Unexpected status code: {response.status_code}, response: {response.text}"
    )

    pagination = response.json().get("pagination")
    assert pagination.get("page") == expected_page, (
        f"Expected page {expected_page}, got {pagination.get('page')}"
    )
    assert pagination.get("page-size") == expected_page_size, (
        f"Expected page size {expected_page_size}, got {pagination.get('page-size')}"
    )

    results = response.json().get(entity_name, [])
    assert len(results) == expected_results_count, (
        f"Expected {expected_results_count} results, got {len(results)}"
    )

    if results:
        first_result_identifier = results[0]["metadata"].get(entity_identifier_name)
        assert first_result_identifier == expected_first_result_name, (
            f"Expected first result identifier '{expected_first_result_name}', got '{first_result_identifier}'"
        )


def _create_project_obj(
    project_name,
    artifact_path,
    source,
    load_source_on_run=False,
    default_function_node_selector=None,
) -> mlrun.common.schemas.Project:
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana",
            source=source,
            load_source_on_run=load_source_on_run,
            goals="some goals",
            artifact_path=artifact_path,
            default_function_node_selector=default_function_node_selector,
        ),
    )
