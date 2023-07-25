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
import random
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.common.schemas

PROJECT_NAME = "my-proj1"
WORKFLOW_NAME = "main"


def test_workflow_does_not_exist(db: Session, client: TestClient):
    _create_proj_with_workflow(client)
    # path with wrong name:
    wrong_name = "not-" + PROJECT_NAME
    resp = client.post(f"projects/{PROJECT_NAME}/workflows/{wrong_name}/submit")
    assert (
        resp.json()["detail"]["reason"] == f"workflow {wrong_name} not found in project"
    )
    assert resp.status_code == HTTPStatus.BAD_REQUEST


def test_bad_schedule_format(db: Session, client: TestClient):
    _create_proj_with_workflow(client)

    # Spec with bad schedule:
    workflow_body = {"spec": {"name": WORKFLOW_NAME, "schedule": "* * 1"}}

    resp = client.post(
        f"projects/{PROJECT_NAME}/workflows/{WORKFLOW_NAME}/submit", json=workflow_body
    )
    assert (
        "Wrong number of fields in crontab expression" in resp.json()["detail"]["error"]
    )
    assert resp.status_code == HTTPStatus.BAD_REQUEST


def test_get_workflow_bad_id(db: Session, client: TestClient):
    _create_proj_with_workflow(client)

    # wrong run id:
    expected_workflow_id = "some id"
    right_id = "".join(random.choices("0123456789abcdef", k=40))
    wrong_id = "".join(random.choices("0123456789abcdef", k=40))
    data = {
        "metadata": {"name": "run-name"},
        "status": {"results": {"workflow_id": expected_workflow_id}},
    }
    mlrun.api.crud.Runs().store_run(db, data, right_id, project=PROJECT_NAME)
    good_resp = client.get(
        f"projects/{PROJECT_NAME}/workflows/{WORKFLOW_NAME}/runs/{right_id}"
    ).json()

    assert (
        good_resp.get("workflow_id", "") == expected_workflow_id
    ), f"response: {good_resp}"
    bad_resp = client.get(
        f"projects/{PROJECT_NAME}/workflows/{WORKFLOW_NAME}/runs/{wrong_id}"
    )
    assert bad_resp.status_code == HTTPStatus.NOT_FOUND


def test_get_workflow_bad_project(db: Session, client: TestClient):
    _create_proj_with_workflow(client)
    # wrong run id:
    expected_workflow_id = "some id"
    run_id = "".join(random.choices("0123456789abcdef", k=40))
    wrong_project_name = "not a project"
    data = {
        "metadata": {"name": "run-name"},
        "status": {"results": {"workflow_id": expected_workflow_id}},
    }
    mlrun.api.crud.Runs().store_run(db, data, run_id, project=PROJECT_NAME)
    resp = client.get(
        f"projects/{wrong_project_name}/workflows/{WORKFLOW_NAME}/runs/{run_id}"
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert f"Run {run_id}:{wrong_project_name} not found" in resp.json()["detail"]


def _create_proj_with_workflow(client: TestClient):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=PROJECT_NAME),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana",
            source="git://github.com/mlrun/project-demo",
            goals="some goals",
            workflows=[{"name": WORKFLOW_NAME}],
        ),
    )
    client.post("projects", json=project.dict())
