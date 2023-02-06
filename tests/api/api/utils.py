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
import uuid
from http import HTTPStatus

import httpx
from fastapi.testclient import TestClient

import mlrun.api.api.endpoints.functions
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.errors

PROJECT = "project-name"


def create_project(client: TestClient, project_name: str = PROJECT, artifact_path=None):
    project = _create_project_obj(project_name, artifact_path)
    resp = client.post("projects", json=project.dict())
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def compile_schedule(schedule_name: str = None, to_json: bool = True):
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


async def create_project_async(
    async_client: httpx.AsyncClient, project_name: str = PROJECT
):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )
    resp = await async_client.post(
        "projects",
        json=project.dict(),
    )
    assert resp.status_code == HTTPStatus.CREATED.value
    return resp


def _create_project_obj(project_name, artifact_path) -> mlrun.api.schemas.Project:
    return mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana",
            source="source",
            goals="some goals",
            artifact_path=artifact_path,
        ),
    )
