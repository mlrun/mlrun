import time
import unittest.mock
from datetime import datetime, timezone
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
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
        "metadata": {"labels": {"kind": mlrun.runtimes.RuntimeKinds.job}},
        "status": {"state": mlrun.runtimes.constants.RunStates.running},
    }
    run_in_progress_uid = "in-progress-uid"
    run_completed = {
        "metadata": {"labels": {"kind": mlrun.runtimes.RuntimeKinds.job}},
        "status": {"state": mlrun.runtimes.constants.RunStates.completed},
    }
    run_completed_uid = "completed-uid"
    run_aborted = {
        "metadata": {"labels": {"kind": mlrun.runtimes.RuntimeKinds.job}},
        "status": {"state": mlrun.runtimes.constants.RunStates.aborted},
    }
    run_aborted_uid = "aborted-uid"
    run_dask = {
        "metadata": {"labels": {"kind": mlrun.runtimes.RuntimeKinds.dask}},
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
    timestamp1 = datetime.now(timezone.utc)

    time.sleep(1)

    timestamp2 = datetime.now(timezone.utc)

    normal_run_1_uid = "normal_run_1_uid"
    normal_run_1 = {
        "metadata": {"uid": normal_run_1_uid},
        "status": {"last_update": timestamp2.isoformat()},
    }
    run = Run(
        uid=normal_run_1_uid,
        project=config.default_project,
        iteration=0,
        start_time=timestamp1,
    )
    run.struct = normal_run_1
    get_db()._upsert(db, run, ignore=True)

    timestamp3 = datetime.now(timezone.utc)

    run_without_last_update_uid = "run_without_last_update_uid"
    run_without_last_update = {
        "metadata": {"uid": run_without_last_update_uid},
        "status": {},
    }
    run = Run(
        uid=run_without_last_update_uid,
        project=config.default_project,
        iteration=0,
        start_time=timestamp3,
    )
    run.struct = run_without_last_update
    get_db()._upsert(db, run, ignore=True)

    timestamp4 = datetime.now(timezone.utc)

    time.sleep(1)

    timestamp5 = datetime.now(timezone.utc)

    normal_run_2_uid = "normal_run_2_uid"
    normal_run_2 = {
        "metadata": {"uid": normal_run_2_uid},
        "status": {"last_update": timestamp5.isoformat()},
    }
    run = Run(
        uid=normal_run_2_uid,
        project=config.default_project,
        iteration=0,
        start_time=timestamp4,
    )
    run.struct = normal_run_2
    get_db()._upsert(db, run, ignore=True)

    # all start time range
    assert_time_range_request(
        client, [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid]
    )
    assert_time_range_request(
        client,
        [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp1.isoformat(),
        start_time_to=timestamp5.isoformat(),
    )
    assert_time_range_request(
        client,
        [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp1.isoformat(),
    )
    assert_time_range_request(
        client,
        [normal_run_1_uid, normal_run_2_uid, run_without_last_update_uid],
        start_time_to=timestamp5.isoformat(),
    )

    # all last update time range (shouldn't contain run_without_last_update)
    assert_time_range_request(
        client,
        [normal_run_1_uid, normal_run_2_uid],
        last_update_time_from=timestamp1,
        last_update_time_to=timestamp5,
    )
    assert_time_range_request(
        client, [normal_run_1_uid, normal_run_2_uid], last_update_time_from=timestamp1,
    )
    assert_time_range_request(
        client, [normal_run_1_uid, normal_run_2_uid], last_update_time_to=timestamp5,
    )

    # catch only first
    assert_time_range_request(
        client,
        [normal_run_1_uid],
        start_time_from=timestamp1,
        start_time_to=timestamp2,
    )
    assert_time_range_request(
        client, [normal_run_1_uid], start_time_to=timestamp2,
    )
    assert_time_range_request(
        client,
        [normal_run_1_uid],
        last_update_time_from=timestamp1,
        last_update_time_to=timestamp2,
    )

    # catch run_without_last_update and last
    assert_time_range_request(
        client,
        [normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp3,
        start_time_to=timestamp5,
    )
    assert_time_range_request(
        client,
        [normal_run_2_uid, run_without_last_update_uid],
        start_time_from=timestamp3,
    )


def assert_time_range_request(client: TestClient, expected_run_uids: list, **filters):
    resp = client.get("/api/runs", params=filters)
    assert resp.status_code == HTTPStatus.OK.value

    runs = resp.json()["runs"]
    assert len(runs) == len(expected_run_uids)
    for run in runs:
        assert run["metadata"]["uid"] in expected_run_uids
