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
import asyncio
import copy
import time
import unittest.mock
import uuid
from datetime import datetime, timedelta, timezone
from http import HTTPStatus

import fastapi
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
from mlrun.config import config
from server.api.db.sqldb.models import Run
from server.api.utils.singletons.db import get_db

RUNS_API_ENDPOINT = "/projects/{project}/runs"


def test_run_with_nan_in_body(db: Session, client: TestClient) -> None:
    """
    This test wouldn't pass if we were using FastAPI default JSONResponse which uses json.dumps to serialize jsons
    It passes only because we changed to use fastapi.responses.ORJSONResponse by default which uses orjson.dumps
    which do handles float("Nan")
    """
    run_with_nan_float = {
        "metadata": {"name": "run-name"},
        "status": {"artifacts": [{"preview": [[0.0, float("Nan"), 1.3]]}]},
    }
    uid = "some-uid"
    project = "some-project"
    server.api.crud.Runs().store_run(db, run_with_nan_float, uid, project=project)
    resp = client.get(f"run/{project}/{uid}")
    assert resp.status_code == HTTPStatus.OK.value


def test_legacy_abort_run(db: Session, client: TestClient) -> None:
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    run_completed = {
        "metadata": {
            "name": "run-name-2",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.completed},
    }
    run_completed_uid = "completed-uid"
    run_aborted = {
        "metadata": {
            "name": "run-name-3",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.aborted},
    }
    run_aborted_uid = "aborted-uid"
    run_dask = {
        "metadata": {
            "name": "run-name-4",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.dask},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_dask_uid = "dask-uid"
    for run, run_uid in [
        (run_in_progress, run_in_progress_uid),
        (run_completed, run_completed_uid),
        (run_aborted, run_aborted_uid),
        (run_dask, run_dask_uid),
    ]:
        server.api.crud.Runs().store_run(db, run, run_uid, project=project)

    runtime_resources = server.api.crud.RuntimeResources()
    runtime_resources.delete_runtime_resources = unittest.mock.Mock()
    abort_body = {"status.state": mlrun.common.runtimes.constants.RunStates.aborted}
    # completed is terminal state - should fail
    response = client.patch(f"run/{project}/{run_completed_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.CONFLICT.value
    # aborted is terminal state - should fail
    response = client.patch(f"run/{project}/{run_aborted_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.CONFLICT.value
    # dask kind not abortable - should fail
    response = client.patch(f"run/{project}/{run_dask_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    # running is ok - should succeed
    response = client.patch(f"run/{project}/{run_in_progress_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.OK.value
    runtime_resources.delete_runtime_resources.assert_called_once()


def test_abort_run(db: Session, client: TestClient) -> None:
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    run_completed = {
        "metadata": {
            "name": "run-name-2",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.completed},
    }
    run_completed_uid = "completed-uid"
    run_aborted = {
        "metadata": {
            "name": "run-name-3",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.aborted},
    }
    run_aborted_uid = "aborted-uid"
    run_dask = {
        "metadata": {
            "name": "run-name-4",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.dask},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_dask_uid = "dask-uid"
    for run, run_uid in [
        (run_in_progress, run_in_progress_uid),
        (run_completed, run_completed_uid),
        (run_aborted, run_aborted_uid),
        (run_dask, run_dask_uid),
    ]:
        server.api.crud.Runs().store_run(db, run, run_uid, project=project)

    abort_body = {
        "status.state": mlrun.common.runtimes.constants.RunStates.aborted,
        "status.error": "Run was aborted by user",
    }
    runtime_resources = server.api.crud.RuntimeResources()
    runtime_resources.delete_runtime_resources = unittest.mock.Mock()
    # completed is terminal state - should fail
    response = client.post(
        f"projects/{project}/runs/{run_completed_uid}/abort", json=abort_body
    )
    assert response.status_code == HTTPStatus.ACCEPTED.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    background_task = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
        db, background_task.metadata.name, project
    )
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
    )
    assert (
        background_task.status.error
        == "Run is already in terminal state, can not be aborted"
    )
    # aborted is terminal state - should fail
    response = client.post(
        f"projects/{project}/runs/{run_aborted_uid}/abort", json=abort_body
    )
    assert response.status_code == HTTPStatus.ACCEPTED.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    background_task = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
        db, background_task.metadata.name, project
    )
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
    )
    assert (
        background_task.status.error
        == "Run is already in terminal state, can not be aborted"
    )
    # dask kind not abortable - should fail
    response = client.post(
        f"projects/{project}/runs/{run_dask_uid}/abort", json=abort_body
    )
    assert response.status_code == HTTPStatus.ACCEPTED.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    background_task = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
        db, background_task.metadata.name, project
    )
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
    )
    assert background_task.status.error == "Run of kind dask can not be aborted"
    # running is ok - should succeed
    response = client.post(
        f"projects/{project}/runs/{run_in_progress_uid}/abort", json=abort_body
    )
    assert response.status_code == HTTPStatus.ACCEPTED.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    background_task = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
        db, background_task.metadata.name, project
    )
    assert (
        background_task.status.state
        == mlrun.common.schemas.BackgroundTaskState.succeeded
    )
    assert background_task.status.error is None
    runtime_resources.delete_runtime_resources.assert_called_once()

    run = server.api.crud.Runs().get_run(db, run_in_progress_uid, 0, project)
    assert run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.aborted
    assert run["status"]["error"] == "Run was aborted by user"


def test_list_runs_times_filters(db: Session, client: TestClient) -> None:
    run_1_start_time = datetime.now(timezone.utc)

    time.sleep(0.1)

    run_1_update_time = datetime.now(timezone.utc)

    run_1_name = "run_1_name"
    run_1_uid = "run_1_uid"
    run_1 = {
        "metadata": {"name": run_1_name, "uid": run_1_uid},
    }
    run = Run(
        name=run_1_name,
        uid=run_1_uid,
        project=config.default_project,
        iteration=0,
        start_time=run_1_start_time,
        updated=run_1_update_time,
    )
    run.struct = run_1
    get_db()._upsert(db, [run], ignore=True)

    between_run_1_and_2 = datetime.now(timezone.utc)

    time.sleep(0.1)

    run_2_start_time = datetime.now(timezone.utc)

    time.sleep(0.1)

    run_2_update_time = datetime.now(timezone.utc)

    run_2_uid = "run_2_uid"
    run_2_name = "run_2_name"
    run_2 = {
        "metadata": {"name": run_2_name, "uid": run_2_uid},
    }
    run = Run(
        name=run_2_name,
        uid=run_2_uid,
        project=config.default_project,
        iteration=0,
        start_time=run_2_start_time,
        updated=run_2_update_time,
    )
    run.struct = run_2
    get_db()._upsert(db, [run], ignore=True)

    # all start time range
    assert_time_range_request(client, [run_1_uid, run_2_uid])
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        start_time_from=run_1_start_time.isoformat(),
        start_time_to=run_2_update_time.isoformat(),
    )
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        start_time_from=run_1_start_time.isoformat(),
    )

    # all last update time range
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        last_update_time_from=run_1_update_time,
        last_update_time_to=run_2_update_time,
    )
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        last_update_time_from=run_1_update_time,
    )
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        last_update_time_to=run_2_update_time,
    )

    # catch only first
    assert_time_range_request(
        client,
        [run_1_uid],
        start_time_from=run_1_start_time,
        start_time_to=between_run_1_and_2,
    )
    assert_time_range_request(
        client,
        [run_1_uid],
        start_time_to=between_run_1_and_2,
    )
    assert_time_range_request(
        client,
        [run_1_uid],
        last_update_time_from=run_1_update_time,
        last_update_time_to=run_2_start_time,
    )

    # catch run_without_last_update and last
    assert_time_range_request(
        client,
        [run_2_uid],
        start_time_from=run_2_start_time,
        start_time_to=run_2_update_time,
    )
    assert_time_range_request(
        client,
        [run_2_uid],
        last_update_time_from=run_2_start_time,
    )


def test_list_runs_partition_by(db: Session, client: TestClient) -> None:
    # Create runs
    projects = ["run-project-1", "run-project-2", "run-project-3"]
    run_names = ["run-name-1", "run-name-2", "run-name-3"]
    suffixes = ["first", "second", "third"]
    iterations = 3
    for project in projects:
        for name in run_names:
            for suffix in suffixes:
                uid = f"{name}-uid-{suffix}"
                for iteration in range(iterations):
                    run = {
                        "metadata": {
                            "name": name,
                            "uid": uid,
                            "project": project,
                            "iter": iteration,
                        },
                    }
                    server.api.crud.Runs().store_run(db, run, uid, iteration, project)

    # basic list, all projects, all iterations so 3 projects * 3 names * 3 uids * 3 iterations = 81
    _list_and_assert_objects(
        client,
        params={},
        expected_number_of_runs=81,
        project="*",
    )

    # basic list, specific project, only iteration 0, so 3 names = 3
    _list_and_assert_objects(
        client,
        params={"iter": False},
        expected_number_of_runs=3,
        project=projects[0],
    )

    # adding start time from to make sure we get all runs (and not just latest)
    # basic list, specific project, only iteration 0, so 3 names * 3 uids = 9
    _list_and_assert_objects(
        client,
        params={
            "iter": False,
            "start_time_from": datetime.now() - timedelta(days=1),
        },
        expected_number_of_runs=9,
        project=projects[0],
    )

    # partioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
    runs = _list_and_assert_objects(
        client,
        params={
            "partition-by": mlrun.common.schemas.RunPartitionByField.project_and_name,
            "partition-sort-by": mlrun.common.schemas.SortField.created,
            "partition-order": mlrun.common.schemas.OrderType.asc,
        },
        expected_number_of_runs=3,
        project=projects[0],
    )
    # sorted by ascending created so only the first ones created
    for run in runs:
        assert "first" in run["metadata"]["uid"]

    # partioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
    runs = _list_and_assert_objects(
        client,
        params={
            "partition-by": mlrun.common.schemas.RunPartitionByField.project_and_name,
            "partition-sort-by": mlrun.common.schemas.SortField.updated,
            "partition-order": mlrun.common.schemas.OrderType.desc,
        },
        expected_number_of_runs=3,
        project=projects[0],
    )
    # sorted by descending updated so only the third ones created
    for run in runs:
        assert "third" in run["metadata"]["uid"]

    # partioned list, specific project, 5 row per partition, so 3 names * 5 row = 15
    _list_and_assert_objects(
        client,
        params={
            "partition-by": mlrun.common.schemas.RunPartitionByField.project_and_name,
            "partition-sort-by": mlrun.common.schemas.SortField.updated,
            "partition-order": mlrun.common.schemas.OrderType.desc,
            "rows-per-partition": 5,
        },
        expected_number_of_runs=15,
        project=projects[0],
    )

    # partitioned list, specific project, 5 rows per partition, max of 2 partitions, so 2 names * 5 rows = 10
    runs = _list_and_assert_objects(
        client,
        params={
            "partition-by": mlrun.common.schemas.RunPartitionByField.project_and_name,
            "partition-sort-by": mlrun.common.schemas.SortField.updated,
            "partition-order": mlrun.common.schemas.OrderType.desc,
            "rows-per-partition": 5,
            "max-partitions": 2,
        },
        expected_number_of_runs=10,
        project=projects[0],
    )
    for run in runs:
        # Partitions are ordered from latest updated to oldest, which means that 3,2 must be here.
        assert run["metadata"]["name"] in ["run-name-2", "run-name-3"]

    # Complex query, with partitioning and filtering over iter==0
    runs = _list_and_assert_objects(
        client,
        params={
            "iter": False,
            "partition-by": mlrun.common.schemas.RunPartitionByField.project_and_name,
            "partition-sort-by": mlrun.common.schemas.SortField.updated,
            "partition-order": mlrun.common.schemas.OrderType.desc,
            "rows-per-partition": 2,
            "max-partitions": 1,
        },
        expected_number_of_runs=2,
        project=projects[0],
    )

    for run in runs:
        assert run["metadata"]["name"] == "run-name-3" and run["metadata"]["iter"] == 0

    # Some negative testing - no sort by field
    response = client.get(
        f"{RUNS_API_ENDPOINT.format(project=projects[0])}?partition-by=name"
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    # An invalid partition-by field - will be failed by fastapi due to schema validation.
    response = client.get(
        f"{RUNS_API_ENDPOINT.format(project=projects[0])}?partition-by=key&partition-sort-by=name"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value


def test_list_runs_single_and_multiple_uids(db: Session, client: TestClient):
    # Create runs
    number_of_runs = 50
    project = "my_project"
    for counter in range(number_of_runs):
        uid = f"uid_{counter}"
        name = f"run_{counter}"
        run = {
            "metadata": {
                "name": name,
                "uid": uid,
                "project": project,
            },
        }
        server.api.crud.Runs().store_run(db, run, uid, project=project)

    runs = _list_and_assert_objects(
        client,
        {
            "uid": "uid_1",
        },
        1,
        project=project,
    )
    assert runs[0]["metadata"]["uid"] == "uid_1"

    uid_list = ["uid_12", "uid_29", "uid_xx", "uid_3"]
    runs = _list_and_assert_objects(
        client,
        {
            "uid": uid_list,
        },
        # One fictive uid
        len(uid_list) - 1,
        project=project,
    )

    expected_uids = set(uid_list)
    for run in runs:
        assert run["metadata"]["uid"] in expected_uids
        expected_uids.remove(run["metadata"]["uid"])


def test_list_runs_with_pagination(db: Session, client: TestClient):
    """
    Test list runs with pagination.
    Create 25 runs, request the first page, then use token to request 2nd and 3rd pages.
    3rd page will contain only 5 runs instead of 10.
    The 4th request with the token will return 404 as the token is now expired.
    Requesting the 4th page without token will return 0 runs.
    """
    # Create runs
    number_of_runs = 25
    project = "my_project"
    for counter in range(number_of_runs):
        uid = f"uid_{counter}"
        name = f"run_{counter}"
        run = {
            "metadata": {
                "name": name,
                "uid": uid,
                "project": project,
            },
        }
        server.api.crud.Runs().store_run(db, run, uid, project=project)

    # Test pagination
    runs, pagination = _list_and_assert_objects(
        client,
        {
            "page": 1,
            "page-size": 10,
            "sort": True,
        },
        10,
        project=project,
    )
    assert pagination["page"] == 1
    assert pagination["page-size"] == 10
    assert runs[0]["metadata"]["name"] == "run_24"

    token = pagination["page-token"]
    runs, pagination = _list_and_assert_objects(
        client,
        {
            "page-token": token,
        },
        10,
        project=project,
    )
    assert pagination["page"] == 2
    assert pagination["page-size"] == 10
    assert runs[0]["metadata"]["name"] == "run_14"

    runs, pagination = _list_and_assert_objects(
        client,
        {
            "page-token": token,
        },
        5,
        project=project,
    )
    assert pagination["page"] == 3
    assert pagination["page-size"] == 10
    assert runs[0]["metadata"]["name"] == "run_4"

    runs = _list_and_assert_objects(
        client,
        {
            "page-token": token,
        },
        0,
        project=project,
    )
    assert not runs

    runs = _list_and_assert_objects(
        client,
        {
            "page": 4,
            "page-size": 10,
            "sort": True,
        },
        0,
        project=project,
    )
    assert not runs


def test_delete_runs_with_permissions(db: Session, client: TestClient):
    server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions = (
        unittest.mock.AsyncMock()
    )

    # delete runs from specific project
    project = "some-project"
    _store_run(db, uid="some-uid", project=project)
    runs = server.api.crud.Runs().list_runs(db, project=project)
    assert len(runs) == 1
    response = client.delete(RUNS_API_ENDPOINT.format(project="*"))
    assert response.status_code == HTTPStatus.OK.value
    runs = server.api.crud.Runs().list_runs(db, project=project)
    assert len(runs) == 0

    # delete runs from all projects
    second_project = "some-project2"
    _store_run(db, uid=None, project=project, name="run-1")
    _store_run(db, uid=None, project=second_project, name="run-2")
    all_runs = server.api.crud.Runs().list_runs(db, project="*")
    assert len(all_runs) == 2
    response = client.delete(RUNS_API_ENDPOINT.format(project="*"))
    assert response.status_code == HTTPStatus.OK.value
    runs = server.api.crud.Runs().list_runs(db, project="*")
    assert len(runs) == 0


def test_delete_runs_without_permissions(db: Session, client: TestClient):
    server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions = (
        unittest.mock.Mock(side_effect=mlrun.errors.MLRunUnauthorizedError())
    )

    # try delete runs with no permission to project (project doesn't contain any runs)
    project = "some-project"
    runs = server.api.crud.Runs().list_runs(db, project=project)
    assert len(runs) == 0
    response = client.delete(RUNS_API_ENDPOINT.format(project=project))
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value

    # try delete runs with no permission to project (project contains runs)
    _store_run(db, uid="some-uid", project=project)
    runs = server.api.crud.Runs().list_runs(db, project=project)
    assert len(runs) == 1
    response = client.delete(RUNS_API_ENDPOINT.format(project=project))
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    runs = server.api.crud.Runs().list_runs(db, project=project)
    assert len(runs) == 1

    # try delete runs from all projects with no permissions
    response = client.delete(RUNS_API_ENDPOINT.format(project="*"))
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    runs = server.api.crud.Runs().list_runs(db, project=project)
    assert len(runs) == 1


def test_store_run_masking(db: Session, client: TestClient, k8s_secrets_mock):
    notifications = [
        {
            "condition": "",
            "kind": "slack",
            "message": "completed",
            "name": "notification-1",
            "secret_params": {
                "webhook": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
                "other_param": "other_value",
            },
            "severity": "info",
            "when": ["error"],
        },
        {
            "condition": "",
            "kind": "slack",
            "message": "completed",
            "name": "notification-1",
            # should not redact secrets
            "params": {
                "secret": "my-secret",
            },
            "severity": "info",
            "when": ["completed"],
        },
    ]

    masked_notifications = copy.deepcopy(notifications)
    masked_notifications[0]["secret_params"]["webhook"] = "REDACTED"
    masked_notifications[0]["secret_params"]["other_param"] = "REDACTED"

    expected_response_params = {
        "spec.notifications": masked_notifications,
    }

    uid = "1234567890"
    project = "test-store-run-masking"
    run = {
        "metadata": {
            "name": "unmasked-run",
            "project": project,
            "uid": uid,
        },
        "spec": {
            "notifications": notifications,
        },
    }

    server.api.crud.Runs().store_run(db, run, uid, project=project)
    resp = client.get(f"run/{project}/{uid}")
    assert resp.status_code == HTTPStatus.OK.value

    response_body = resp.json()["data"]
    for param, expected_value in expected_response_params.items():
        value = mlrun.utils.get_in(response_body, param)
        assert value == expected_value


def test_abort_run_already_in_progress(db: Session, client: TestClient) -> None:
    project = "some-project"
    background_task_name = "background-task-name"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {
            "state": mlrun.common.runtimes.constants.RunStates.aborting,
            "abort_task_id": background_task_name,
        },
    }
    run_in_progress_uid = "in-progress-uid"
    server.api.crud.Runs().store_run(
        db, run_in_progress, run_in_progress_uid, project=project
    )

    # mock abortion already in progress
    server.api.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task(
        db,
        project,
        fastapi.BackgroundTasks(),
        asyncio.sleep,
        timeout=100,
        name=background_task_name,
        delay=5,
    )

    # abort again should return the same background task
    response = client.post(
        f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
    )
    assert response.status_code == HTTPStatus.ACCEPTED.value
    background_task = mlrun.common.schemas.BackgroundTask(**response.json())
    assert (
        background_task.status.state == mlrun.common.schemas.BackgroundTaskState.running
    )
    assert background_task.metadata.name == background_task_name


def test_abort_aborted_run_with_background_task(
    db: Session, client: TestClient
) -> None:
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    server.api.crud.Runs().store_run(
        db, run_in_progress, run_in_progress_uid, project=project
    )

    with unittest.mock.patch.object(
        server.api.crud.RuntimeResources, "delete_runtime_resources"
    ):
        response = client.post(
            f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value
        background_task_1 = mlrun.common.schemas.BackgroundTask(**response.json())
        background_task_1 = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
            db, background_task_1.metadata.name, project
        )
        assert (
            background_task_1.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )
        run = server.api.crud.Runs().get_run(db, run_in_progress_uid, 0, project)
        assert (
            run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.aborted
        )
        assert run["status"]["abort_task_id"] == background_task_1.metadata.name

        # abort again should return the same background task
        response = client.post(
            f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value
        background_task_2 = mlrun.common.schemas.BackgroundTask(**response.json())
        background_task_2 = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
            db, background_task_2.metadata.name, project
        )
        assert (
            background_task_2.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )
        assert background_task_2.metadata.name == background_task_1.metadata.name


def test_abort_aborted_run_passed_grace_period(db: Session, client: TestClient) -> None:
    mlrun.mlconf.background_tasks.default_timeouts.operations.abort_grace_period = 0
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    server.api.crud.Runs().store_run(
        db, run_in_progress, run_in_progress_uid, project=project
    )

    with unittest.mock.patch.object(
        server.api.crud.RuntimeResources, "delete_runtime_resources"
    ):
        # abort once
        response = client.post(
            f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value
        background_task_1 = mlrun.common.schemas.BackgroundTask(**response.json())
        background_task_1 = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
            db, background_task_1.metadata.name, project
        )
        assert (
            background_task_1.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )
        run = server.api.crud.Runs().get_run(db, run_in_progress_uid, 0, project)
        assert (
            run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.aborted
        )
        assert run["status"]["abort_task_id"] == background_task_1.metadata.name

        # abort again should create a new failed background task
        response = client.post(
            f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value
        background_task_2 = mlrun.common.schemas.BackgroundTask(**response.json())
        background_task_2 = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
            db, background_task_2.metadata.name, project
        )
        assert (
            background_task_2.status.state
            == mlrun.common.schemas.BackgroundTaskState.failed
        )
        assert background_task_2.metadata.name != background_task_1.metadata.name
        assert (
            background_task_2.status.error
            == "Run is already in terminal state, can not be aborted"
        )


def test_abort_run_background_task_not_found(db: Session, client: TestClient) -> None:
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {
            "state": mlrun.common.runtimes.constants.RunStates.aborting,
            # add a background task id that doesn't exist
            "abort_task_id": "background-task-name",
        },
    }
    run_in_progress_uid = "in-progress-uid"
    server.api.crud.Runs().store_run(
        db, run_in_progress, run_in_progress_uid, project=project
    )

    with unittest.mock.patch.object(
        server.api.crud.RuntimeResources, "delete_runtime_resources"
    ):
        response = client.post(
            f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value
        background_task_1 = mlrun.common.schemas.BackgroundTask(**response.json())
        background_task_1 = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
            db, background_task_1.metadata.name, project
        )
        assert (
            background_task_1.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )
        run = server.api.crud.Runs().get_run(db, run_in_progress_uid, 0, project)
        assert (
            run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.aborted
        )
        assert run["status"]["abort_task_id"] == background_task_1.metadata.name


def test_abort_aborted_run_failure(db: Session, client: TestClient) -> None:
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.common.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    server.api.crud.Runs().store_run(
        db, run_in_progress, run_in_progress_uid, project=project
    )

    with unittest.mock.patch.object(
        server.api.crud.RuntimeResources,
        "delete_runtime_resources",
        side_effect=Exception("some error"),
    ):
        response = client.post(
            f"projects/{project}/runs/{run_in_progress_uid}/abort", json={}
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value
        background_task = mlrun.common.schemas.BackgroundTask(**response.json())
        background_task = server.api.utils.background_tasks.ProjectBackgroundTasksHandler().get_background_task(
            db, background_task.metadata.name, project
        )
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.failed
        )
        assert background_task.status.error == "some error"


def _store_run(db, uid, project="some-project", name="run-name"):
    run_with_nan_float = {
        "metadata": {"name": name},
        "status": {"artifacts": [{"preview": [[0.0, float("Nan"), 1.3]]}]},
    }
    if not uid:
        uid = str(uuid.uuid4())
    return server.api.crud.Runs().store_run(
        db, run_with_nan_float, uid, project=project
    )


def _list_and_assert_objects(
    client: TestClient, params, expected_number_of_runs: int, project: str
):
    response = client.get(RUNS_API_ENDPOINT.format(project=project), params=params)
    assert response.status_code == HTTPStatus.OK.value, response.text

    response_json = response.json()
    runs = response.json()["runs"]
    assert len(runs) == expected_number_of_runs

    if (
        "pagination" in response_json
        and response_json["pagination"]
        and response_json["pagination"]["page"]
    ):
        return runs, response_json["pagination"]

    return runs


def assert_time_range_request(client: TestClient, expected_run_uids: list, **filters):
    resp = client.get("runs", params=filters)
    assert resp.status_code == HTTPStatus.OK.value

    runs = resp.json()["runs"]
    assert len(runs) == len(expected_run_uids)
    for run in runs:
        assert run["metadata"]["uid"] in expected_run_uids
