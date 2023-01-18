import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm.session

import mlrun.api.crud
import mlrun.api.main
import mlrun.api.utils.clients.log_collector
import mlrun.api.utils.singletons.db
from tests.api.utils.clients.test_log_collector import StartLogResponse


class TestCollectRunSLogs:
    @pytest.mark.asyncio
    async def test_collect_runs_with_runs(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db, project_name, uid=run_uid, name=run_uid, iteration=i, kind="job"
                )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            project="*",
            requested_logs=False,
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=StartLogResponse(True, "")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._collect_runs_logs()

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 1
        )
        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_args[
                1
            ]["uids"]
            == run_uids
        )

    @pytest.mark.asyncio
    async def test_collect_runs_with_no_runs(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            project="*",
            requested_logs=False,
            only_uids=False,
        )
        assert len(runs) == 0

        log_collector._call = unittest.mock.AsyncMock(
            return_value=StartLogResponse(True, "")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._collect_runs_logs()

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 0
        )

    @pytest.mark.asyncio
    async def test_collect_logs_with_runs_fails(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db, project_name, uid=run_uid, name=run_uid, iteration=i, kind="job"
                )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            project="*",
            requested_logs=False,
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=StartLogResponse(False, "some error")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._collect_runs_logs()

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 0
        )

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_local_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=StartLogResponse(True, "")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project")
        run_uid = await mlrun.api.main._start_log_for_run(run)
        assert run_uid == uid
        assert log_collector._call.call_count == 0

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_job_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=StartLogResponse(True, "")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project", kind="job")
        run_uid = await mlrun.api.main._start_log_for_run(run)
        assert run_uid == uid
        assert log_collector._call.call_count == 1

    @pytest.mark.asyncio
    async def test_start_log_for_run_failure(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=StartLogResponse(False, "some error")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project", kind="job")
        run_uid = await mlrun.api.main._start_log_for_run(run)
        assert run_uid is None
        assert log_collector._call.call_count == 1

    @pytest.mark.asyncio
    async def test_fufu_real_log_collector(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        _, _, uid, _, run = _create_new_run(
            db,
            "default",
            uid="cde099c6724742859b8b2115eb767429",
            name="handler",
            kind="job",
            store=False,
        )
        run_uid = await mlrun.api.main._start_log_for_run(run)
        assert run_uid == uid

        logs = await mlrun.api.utils.clients.log_collector.LogCollectorClient().get_logs(
            run_uid=uid, project="default", offset=0, size=-1
        )
        # assert len(logs) < 95000

        logs = await mlrun.api.utils.clients.log_collector.LogCollectorClient().get_logs(
            run_uid=uid, project="default", offset=95000, size=-1
        )
        assert len(logs) > 0



def _create_new_run(
    db_session: sqlalchemy.orm.session.Session,
    project="project",
    name="run-name-1",
    uid="run-uid",
    iteration=0,
    kind="",
    state=mlrun.runtimes.constants.RunStates.created,
    store: bool = True,
):
    labels = {"kind": kind}
    run = {
        "metadata": {
            "name": name,
            "uid": uid,
            "project": project,
            "iter": iteration,
            "labels": labels,
        },
        "status": {"state": state},
    }

    mlrun.api.crud.Runs().store_run(
        db_session, run, uid, iter=iteration, project=project
    )
    return project, name, uid, iteration, run
