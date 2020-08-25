from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api.utils.singletons.db import get_db


def test_run_with_nan_in_body(db: Session, client: TestClient, monkeypatch) -> None:
    run_with_nan_float = {
        "status": {"artifacts": [{"preview": [[0.0, float("Nan"), 1.3]]}]},
    }
    uid = "some-uid"
    project = "some-project"
    get_db().store_run(db, run_with_nan_float, uid, project)
    resp = client.get(f"/api/run/{project}/{uid}")
    assert resp.status_code == HTTPStatus.OK.value
