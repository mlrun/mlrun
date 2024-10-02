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
import unittest.mock
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import server.api.api.endpoints.workflows
import server.api.crud

PROJECT_NAME = "my-proj1"
WORKFLOW_NAME = "main"


def test_workflow_does_not_exist(db: Session, client: TestClient):
    _create_proj_with_workflow(client)
    # path with wrong name:
    wrong_name = "not-" + PROJECT_NAME
    resp = client.post(f"projects/{PROJECT_NAME}/workflows/{wrong_name}/submit")
    assert (
        resp.json()["detail"]["reason"] == f"Workflow {wrong_name} not found in project"
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


def test_get_workflow_fail_fast(db: Session, client: TestClient):
    _create_proj_with_workflow(client)

    right_id = "".join(random.choices("0123456789abcdef", k=40))
    data = {
        "metadata": {
            "name": "run-name",
            "labels": {
                mlrun_constants.MLRunInternalLabels.job_type: "workflow-runner",
            },
        },
        "spec": {
            "parameters": {"workflow_name": "main"},
        },
        "status": {
            "state": "failed",
            "error": "some dummy error",
            # workflow id is empty to simulate a failed remote runner
            "results": {"workflow_id": None},
        },
    }
    server.api.crud.Runs().store_run(db, data, right_id, project=PROJECT_NAME)
    resp = client.get(
        f"projects/{PROJECT_NAME}/workflows/{WORKFLOW_NAME}/runs/{right_id}"
    )

    # remote runner has failed, so the run should be failed
    assert resp.status_code == HTTPStatus.PRECONDITION_FAILED
    assert "some dummy error" in resp.json()["detail"]


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
    server.api.crud.Runs().store_run(db, data, right_id, project=PROJECT_NAME)
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
    server.api.crud.Runs().store_run(db, data, run_id, project=PROJECT_NAME)
    resp = client.get(
        f"projects/{wrong_project_name}/workflows/{WORKFLOW_NAME}/runs/{run_id}"
    )
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert (
        f"Run uid {run_id} of project {wrong_project_name} not found"
        in resp.json()["detail"]
    )


def test_schedule_not_enriched(db: Session, client: TestClient, k8s_secrets_mock):
    _create_proj_with_workflow(client, schedule="* * * * 1")

    # Spec with bad schedule:
    workflow_body = {"spec": {"name": WORKFLOW_NAME}}

    class UIDMock:
        def uid(self):
            return "some uid"

    with unittest.mock.patch.object(
        server.api.crud.WorkflowRunners, "run", return_value=UIDMock()
    ):
        resp = client.post(
            f"projects/{PROJECT_NAME}/workflows/{WORKFLOW_NAME}/submit",
            json=workflow_body,
        )
        assert resp.status_code == HTTPStatus.ACCEPTED
        response_data = resp.json()
        assert response_data["schedule"] is None


def test_fill_workflow_missing_fields_preserves_empty_node_selector(
    db: Session, client: TestClient
):
    # Test to ensure that the `_fill_workflow_missing_fields_from_project` function, called as part of the `submit`
    # workflow endpoint, preserves an empty value in the workflow runner node selector when passed within the workflow
    # spec. Preserving empty values is important because they indicate that a specific node selector should be removed
    # or cleared. Therefore, these empty values must remain in the resulting workflow spec after processing.
    project = _create_proj_with_workflow(client)
    workflow_runner_node_selector = {"test-ns": ""}
    workflow_spec = mlrun.common.schemas.WorkflowSpec(
        name=WORKFLOW_NAME, workflow_runner_node_selector=workflow_runner_node_selector
    )
    res_workflow = (
        server.api.api.endpoints.workflows._fill_workflow_missing_fields_from_project(
            project=project,
            workflow_name=WORKFLOW_NAME,
            spec=workflow_spec,
            arguments={},
        )
    )
    assert res_workflow.workflow_runner_node_selector == workflow_runner_node_selector


def _create_proj_with_workflow(client: TestClient, **extra_workflow_spec):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=PROJECT_NAME),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana",
            source="git://github.com/mlrun/project-demo",
            goals="some goals",
            workflows=[{"name": WORKFLOW_NAME, **extra_workflow_spec}],
        ),
    )
    client.post("projects", json=project.dict())
    return project
