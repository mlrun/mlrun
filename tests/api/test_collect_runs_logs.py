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
import asyncio
import unittest.mock

import deepdiff
import fastapi.testclient
import pytest
import sqlalchemy.orm.session

import mlrun.api.crud
import mlrun.api.main
import mlrun.api.utils.clients.log_collector
import mlrun.api.utils.singletons.db
from tests.api.utils.clients.test_log_collector import BaseLogCollectorResponse


class TestCollectRunSLogs:
    def setup_method(self):
        # setting semaphore for each setup to ensure event loop is already running
        self.start_log_limit = asyncio.Semaphore(1)

    @pytest.mark.asyncio
    async def test_collect_logs_with_runs(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db, project_name, uid=run_uid, name=run_uid, iteration=i, kind="job"
                )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._initiate_logs_collection(self.start_log_limit)

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 1
        )
        assert (
            deepdiff.DeepDiff(
                mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_args[
                    1
                ][
                    "uids"
                ],
                run_uids,
                ignore_order=True,
            )
            == {}
        )

    @pytest.mark.asyncio
    async def test_collect_logs_with_no_runs(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 0

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._initiate_logs_collection(self.start_log_limit)

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 0
        )

    @pytest.mark.asyncio
    async def test_collect_logs_on_startup(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db,
                    project_name,
                    uid=run_uid,
                    name=run_uid,
                    iteration=i,
                    kind="job",
                    state=mlrun.runtimes.constants.RunStates.completed,
                )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._verify_log_collection_started_on_startup(
            self.start_log_limit
        )

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 1
        )
        assert (
            deepdiff.DeepDiff(
                mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_args[
                    1
                ][
                    "uids"
                ],
                run_uids,
                ignore_order=True,
            )
            == {}
        )

    @pytest.mark.asyncio
    async def test_collect_logs_with_runs_fails(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db, project_name, uid=run_uid, name=run_uid, iteration=i, kind="job"
                )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "some error")
        )
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await mlrun.api.main._initiate_logs_collection(self.start_log_limit)

        assert (
            mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 0
        )

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_local_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project")
        run_uid = await mlrun.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid == uid
        assert log_collector._call.call_count == 0

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_job_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project", kind="job")
        run_uid = await mlrun.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid == uid
        assert log_collector._call.call_count == 1

    @pytest.mark.asyncio
    async def test_start_log_for_run_failure(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.get_log_collector_client()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "some error")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project", kind="job")
        run_uid = await mlrun.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid is None
        assert log_collector._call.call_count == 1

    @pytest.mark.asyncio
    async def test_stop_logs(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )

        # create a mock runs list
        num_of_runs, num_of_projects = 3, 2
        runs, run_uids = [], []

        for i in range(num_of_projects):
            project_name = f"some-project-{i}"
            for j in range(num_of_runs):
                run_uid = f"some-uid-{j}"
                runs.append(
                    {
                        "metadata": {
                            "project": project_name,
                            "uid": run_uid,
                        }
                    }
                )
                if run_uid not in run_uids:
                    run_uids.append(run_uid)

        await mlrun.api.main._stop_logs_for_runs(runs)

        assert log_collector._call.call_count == num_of_projects

        stop_log_request = log_collector._call.call_args[0][1]

        # verify that the stop log request is correct, with the last project name
        assert stop_log_request.project == f"some-project-{num_of_projects-1}"
        assert len(stop_log_request.runUIDs) == num_of_runs
        assert (
            deepdiff.DeepDiff(
                list(stop_log_request.runUIDs),
                run_uids,
                ignore_order=True,
            )
            == {}
        )

    @pytest.mark.asyncio
    async def test_verify_stop_logs_on_startup(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            _create_new_run(
                db,
                project_name,
                uid=run_uid,
                name=run_uid,
                kind="job",
                state=mlrun.runtimes.constants.RunStates.completed,
            )

        # update requested logs field to True
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs(
            db, run_uids, True
        )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[True],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(return_value=None)

        await mlrun.api.main._verify_log_collection_stopped_on_startup()

        assert log_collector._call.call_count == 1
        assert log_collector._call.call_args[0][0] == "StopLog"
        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name
        assert len(stop_log_request.runUIDs) == 3
        assert (
            deepdiff.DeepDiff(
                list(stop_log_request.runUIDs),
                run_uids,
                ignore_order=True,
            )
            == {}
        )

        # update requested logs field to False for one run
        mlrun.api.utils.singletons.db.get_db().update_runs_requested_logs(
            db, [run_uids[0]], False
        )

        runs = mlrun.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[True],
            only_uids=False,
        )
        assert len(runs) == 2

        await mlrun.api.main._verify_log_collection_stopped_on_startup()

        assert log_collector._call.call_count == 2
        assert log_collector._call.call_args[0][0] == "StopLog"
        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name
        assert len(stop_log_request.runUIDs) == 2
        assert (
            deepdiff.DeepDiff(
                list(stop_log_request.runUIDs),
                run_uids[1:],
                ignore_order=True,
            )
            == {}
        )


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
    if store:
        mlrun.api.crud.Runs().store_run(
            db_session, run, uid, iter=iteration, project=project
        )
    return project, name, uid, iteration, run
