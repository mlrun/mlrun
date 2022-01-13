import time
import unittest.mock
from datetime import datetime, timezone
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.errors
import mlrun.runtimes.constants
from mlrun.api.db.sqldb.models import Run
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config


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
    mlrun.api.crud.Runs().store_run(db, run_with_nan_float, uid, project=project)
    resp = client.get(f"/api/run/{project}/{uid}")
    assert resp.status_code == HTTPStatus.OK.value


def test_abort_run(db: Session, client: TestClient) -> None:
    project = "some-project"
    run_in_progress = {
        "metadata": {
            "name": "run-name-1",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    run_completed = {
        "metadata": {
            "name": "run-name-2",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.runtimes.constants.RunStates.completed},
    }
    run_completed_uid = "completed-uid"
    run_aborted = {
        "metadata": {
            "name": "run-name-3",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.job},
        },
        "status": {"state": mlrun.runtimes.constants.RunStates.aborted},
    }
    run_aborted_uid = "aborted-uid"
    run_dask = {
        "metadata": {
            "name": "run-name-4",
            "labels": {"kind": mlrun.runtimes.RuntimeKinds.dask},
        },
        "status": {"state": mlrun.runtimes.constants.RunStates.running},
    }
    run_dask_uid = "dask-uid"
    for run, run_uid in [
        (run_in_progress, run_in_progress_uid),
        (run_completed, run_completed_uid),
        (run_aborted, run_aborted_uid),
        (run_dask, run_dask_uid),
    ]:
        mlrun.api.crud.Runs().store_run(db, run, run_uid, project=project)

    mlrun.api.crud.RuntimeResources().delete_runtime_resources = unittest.mock.Mock()
    abort_body = {"status.state": mlrun.runtimes.constants.RunStates.aborted}
    # completed is terminal state - should fail
    response = client.patch(f"/api/run/{project}/{run_completed_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.CONFLICT.value
    # aborted is terminal state - should fail
    response = client.patch(f"/api/run/{project}/{run_aborted_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.CONFLICT.value
    # dask kind not abortable - should fail
    response = client.patch(f"/api/run/{project}/{run_dask_uid}", json=abort_body)
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    # running is ok - should succeed
    response = client.patch(
        f"/api/run/{project}/{run_in_progress_uid}", json=abort_body
    )
    assert response.status_code == HTTPStatus.OK.value
    mlrun.api.crud.RuntimeResources().delete_runtime_resources.assert_called_once()


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
    get_db()._upsert(db, run, ignore=True)

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
    get_db()._upsert(db, run, ignore=True)

    # all start time range
    assert_time_range_request(client, [run_1_uid, run_2_uid])
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        start_time_from=run_1_start_time.isoformat(),
        start_time_to=run_2_update_time.isoformat(),
    )
    assert_time_range_request(
        client, [run_1_uid, run_2_uid], start_time_from=run_1_start_time.isoformat(),
    )

    # all last update time range
    assert_time_range_request(
        client,
        [run_1_uid, run_2_uid],
        last_update_time_from=run_1_update_time,
        last_update_time_to=run_2_update_time,
    )
    assert_time_range_request(
        client, [run_1_uid, run_2_uid], last_update_time_from=run_1_update_time,
    )
    assert_time_range_request(
        client, [run_1_uid, run_2_uid], last_update_time_to=run_2_update_time,
    )

    # catch only first
    assert_time_range_request(
        client,
        [run_1_uid],
        start_time_from=run_1_start_time,
        start_time_to=between_run_1_and_2,
    )
    assert_time_range_request(
        client, [run_1_uid], start_time_to=between_run_1_and_2,
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
        client, [run_2_uid], last_update_time_from=run_2_start_time,
    )


def test_list_runs_partition_by(db: Session, client: TestClient) -> None:
    # Create runs
    projects = ["run-project-1", "run-project-2", "run-project-3"]
    run_names = ["run-name-1", "run-name-2", "run-name-3"]
    for project in projects:
        for name in run_names:
            for suffix in ["first", "second", "third"]:
                uid = f"{name}-uid-{suffix}"
                for iteration in range(3):
                    run = {
                        "metadata": {
                            "name": name,
                            "uid": uid,
                            "project": project,
                            "iter": iteration,
                        },
                    }
                    mlrun.api.crud.Runs().store_run(db, run, uid, iteration, project)

    # basic list, all projects, all iterations so 3 projects * 3 names * 3 uids * 3 iterations = 81
    runs = _list_and_assert_objects(client, {"project": "*"}, 81,)

    # basic list, specific project, only iteration 0, so 3 names * 3 uids = 9
    runs = _list_and_assert_objects(client, {"project": projects[0], "iter": False}, 9,)

    # partioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
    runs = _list_and_assert_objects(
        client,
        {
            "project": projects[0],
            "partition-by": mlrun.api.schemas.RunPartitionByField.name,
            "partition-sort-by": mlrun.api.schemas.SortField.created,
            "partition-order": mlrun.api.schemas.OrderType.asc,
        },
        3,
    )
    # sorted by ascending created so only the first ones created
    for run in runs:
        assert "first" in run["metadata"]["uid"]

    # partioned list, specific project, 1 row per partition by default, so 3 names * 1 row = 3
    runs = _list_and_assert_objects(
        client,
        {
            "project": projects[0],
            "partition-by": mlrun.api.schemas.RunPartitionByField.name,
            "partition-sort-by": mlrun.api.schemas.SortField.updated,
            "partition-order": mlrun.api.schemas.OrderType.desc,
        },
        3,
    )
    # sorted by descending updated so only the third ones created
    for run in runs:
        assert "third" in run["metadata"]["uid"]

    # partioned list, specific project, 5 row per partition, so 3 names * 5 row = 15
    runs = _list_and_assert_objects(
        client,
        {
            "project": projects[0],
            "partition-by": mlrun.api.schemas.RunPartitionByField.name,
            "partition-sort-by": mlrun.api.schemas.SortField.updated,
            "partition-order": mlrun.api.schemas.OrderType.desc,
            "rows-per-partition": 5,
        },
        15,
    )

    # Some negative testing - no sort by field
    response = client.get("/api/runs?partition-by=name")
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    # An invalid partition-by field - will be failed by fastapi due to schema validation.
    response = client.get("/api/runs?partition-by=key&partition-sort-by=name")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value


def _list_and_assert_objects(client: TestClient, params, expected_number_of_runs: int):
    response = client.get("/api/runs", params=params)
    assert response.status_code == HTTPStatus.OK.value, response.text

    runs = response.json()["runs"]
    assert len(runs) == expected_number_of_runs
    return runs


def assert_time_range_request(client: TestClient, expected_run_uids: list, **filters):
    resp = client.get("/api/runs", params=filters)
    assert resp.status_code == HTTPStatus.OK.value

    runs = resp.json()["runs"]
    assert len(runs) == len(expected_run_uids)
    for run in runs:
        assert run["metadata"]["uid"] in expected_run_uids
