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
#
import uuid
from http import HTTPStatus

import httpx
from fastapi.testclient import TestClient

import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.common.schemas
import mlrun.errors

PROJECT = "project-name"


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


def compile_schedule(schedule_name: str = None, to_json: bool = True):
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
    assert response.status_code == HTTPStatus.OK.value

    pagination = response.json().get("pagination")
    assert (
        pagination.get("page") == expected_page
    ), f"Expected page {expected_page}, got {pagination.get('page')}"
    assert (
        pagination.get("page-size") == expected_page_size
    ), f"Expected page size {expected_page_size}, got {pagination.get('page-size')}"

    results = response.json().get(entity_name, [])
    assert (
        len(results) == expected_results_count
    ), f"Expected {expected_results_count} results, got {len(results)}"

    if results:
        first_result_identifier = results[0]["metadata"].get(entity_identifier_name)
        assert (
            first_result_identifier == expected_first_result_name
        ), f"Expected first result identifier '{expected_first_result_name}', got '{first_result_identifier}'"


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
