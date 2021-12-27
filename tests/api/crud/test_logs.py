import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.crud


def test_log(db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient):
    project = "project-name"
    uid = "m33"
    data1, data2 = b"ab", b"cd"
    mlrun.api.crud.Runs().store_run(
        db, {"some-run-data": "blabla"}, uid, project=project
    )
    mlrun.api.crud.Logs().store_log(data1, project, uid)
    _, log = mlrun.api.crud.Logs().get_logs(db, project, uid)
    assert data1 == log, "get log 1"

    mlrun.api.crud.Logs().store_log(data2, project, uid, append=True)
    _, log = mlrun.api.crud.Logs().get_logs(db, project, uid)
    assert data1 + data2 == log, "get log 2"

    mlrun.api.crud.Logs().store_log(data1, project, uid, append=False)
    _, log = mlrun.api.crud.Logs().get_logs(db, project, uid)
    assert data1 == log, "get log append=False"
