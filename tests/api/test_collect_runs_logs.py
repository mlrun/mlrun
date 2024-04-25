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
import asyncio
import time
import unittest.mock

import deepdiff
import fastapi.testclient
import pytest
import sqlalchemy.orm.session

import mlrun.config
import server.api.crud
import server.api.main
import server.api.utils.clients.log_collector
import server.api.utils.singletons.db
from tests.api.utils.clients.test_log_collector import (
    BaseLogCollectorResponse,
    ListRunsResponse,
)


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
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db, project_name, uid=run_uid, name=run_uid, iteration=i, kind="job"
                )

        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        server.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await server.api.main._initiate_logs_collection(self.start_log_limit)

        assert (
            server.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 1
        )
        assert (
            deepdiff.DeepDiff(
                server.api.utils.singletons.db.get_db().update_runs_requested_logs.call_args[
                    1
                ]["uids"],
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
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 0

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        server.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await server.api.main._initiate_logs_collection(self.start_log_limit)

        assert (
            server.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 0
        )

    @pytest.mark.asyncio
    async def test_collect_logs_on_startup(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

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

        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        server.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await server.api.main._verify_log_collection_started_on_startup(
            self.start_log_limit
        )

        assert (
            server.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 1
        )
        assert (
            deepdiff.DeepDiff(
                server.api.utils.singletons.db.get_db().update_runs_requested_logs.call_args[
                    1
                ]["uids"],
                run_uids,
                ignore_order=True,
            )
            == {}
        )

    @pytest.mark.asyncio
    async def test_collect_logs_for_old_runs_on_startup(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
        monkeypatch,
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        new_uid = "new_uid"
        old_uid = "old_uid"

        # create first run
        _create_new_run(
            db,
            project_name,
            uid=old_uid,
            name=old_uid,
            kind="job",
            state=mlrun.runtimes.constants.RunStates.completed,
        )

        # sleep for 3 seconds to make sure the runs are not created at the same time
        time.sleep(3)

        # create second run
        _create_new_run(
            db,
            project_name,
            uid=new_uid,
            name=new_uid,
            kind="job",
            state=mlrun.runtimes.constants.RunStates.completed,
        )

        # verify that we have 2 runs
        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 2

        # change mlrun config so that the old run will be considered as old
        mlrun.mlconf.runtime_resources_deletion_grace_period = 2

        log_collector_call_mock = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        monkeypatch.setattr(log_collector, "_call", log_collector_call_mock)
        update_runs_requested_logs_mock = unittest.mock.Mock()
        monkeypatch.setattr(
            server.api.utils.singletons.db.get_db(),
            "update_runs_requested_logs",
            update_runs_requested_logs_mock,
        )

        await server.api.main._verify_log_collection_started_on_startup(
            self.start_log_limit
        )

        assert update_runs_requested_logs_mock.call_count == 1
        assert len(update_runs_requested_logs_mock.call_args[1]["uids"]) == 1
        assert update_runs_requested_logs_mock.call_args[1]["uids"][0] == new_uid

    @pytest.mark.asyncio
    async def test_collect_logs_consecutive_failures(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
        monkeypatch,
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        success_uid = "success_uid"
        failure_uid = "failure_uid"

        for run_uid in [success_uid, failure_uid]:
            _create_new_run(
                db,
                project_name,
                uid=run_uid,
                name=run_uid,
                kind="job",
                state=mlrun.runtimes.constants.RunStates.completed,
            )

        # verify that we have 2 runs
        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 2

        # change the max_consecutive_start_log_requests to 2, as per the following calculation:
        # max_consecutive_start_log_requests = int(
        #         int(config.log_collector.failed_runs_grace_period)
        #         / int(config.log_collector.periodic_start_log_interval)
        #     )
        mlrun.mlconf.log_collector.failed_runs_grace_period = 20
        mlrun.mlconf.log_collector.periodic_start_log_interval = 10

        log_collector_call_mock = unittest.mock.AsyncMock(
            side_effect=[
                # failure response for the first call (failure_uid)
                BaseLogCollectorResponse(False, "some error"),
                # success response for the second call (success_uid)
                BaseLogCollectorResponse(True, ""),
                # failure response for the third call (failure_uid)
                BaseLogCollectorResponse(False, "some error"),
                BaseLogCollectorResponse(False, "some error"),
            ]
        )
        monkeypatch.setattr(log_collector, "_call", log_collector_call_mock)
        update_runs_requested_logs_mock = unittest.mock.Mock()
        monkeypatch.setattr(
            server.api.utils.singletons.db.get_db(),
            "update_runs_requested_logs",
            update_runs_requested_logs_mock,
        )

        for i in range(3):
            await server.api.main._initiate_logs_collection(self.start_log_limit)

        assert update_runs_requested_logs_mock.call_count == 2
        # verify that `failure_uid` is also updated in the second call
        assert failure_uid in update_runs_requested_logs_mock.call_args[1]["uids"]

    @pytest.mark.asyncio
    async def test_collect_logs_with_runs_fails(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"
        run_uids = ["some_uid", "some_uid2", "some_uid3"]
        for run_uid in run_uids:
            for i in range(3):
                _create_new_run(
                    db, project_name, uid=run_uid, name=run_uid, iteration=i, kind="job"
                )

        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[False],
            only_uids=False,
        )
        assert len(runs) == 3

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "some error")
        )
        server.api.utils.singletons.db.get_db().update_runs_requested_logs = (
            unittest.mock.Mock()
        )

        await server.api.main._initiate_logs_collection(self.start_log_limit)

        assert (
            server.api.utils.singletons.db.get_db().update_runs_requested_logs.call_count
            == 0
        )

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_local_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project")
        run_uid = await server.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid == uid
        assert log_collector._call.call_count == 0

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_job_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project", kind="job")
        run_uid = await server.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid == uid
        assert log_collector._call.call_count == 1

    @pytest.mark.asyncio
    async def test_start_log_for_run_success_dask_kind(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        project_name = "some-project"
        uid = "my-uid"
        function_name = "some-function"
        function = f"{project_name}/{function_name}@{uid}"

        _, _, uid, _, run = _create_new_run(
            db, "some-project", kind="dask", function=function
        )
        run_uid = await server.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid == uid
        # not expected to call start log, because dask is not log collectable runtime
        assert log_collector._call.call_count == 0

    @pytest.mark.asyncio
    async def test_start_log_for_run_failure(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "some error")
        )
        _, _, uid, _, run = _create_new_run(db, "some-project", kind="job")
        run_uid = await server.api.main._start_log_for_run(
            run, self.start_log_limit, raise_on_error=False
        )
        assert run_uid is None
        assert log_collector._call.call_count == 1

    @pytest.mark.asyncio
    async def test_stop_logs(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )

        # create a mock runs list
        num_of_runs, num_of_projects = 1000, 2
        runs, run_uids = [], []
        stop_logs_run_uids_chunk_size = 200

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

        await server.api.main._stop_logs_for_runs(
            runs, chunk_size=stop_logs_run_uids_chunk_size
        )

        # every time we stop <stop_logs_run_uids_chunk_size> amoutn of runs
        assert log_collector._call.call_count == num_of_projects * (
            num_of_runs / stop_logs_run_uids_chunk_size
        )

        stop_log_request = log_collector._call.call_args_list[0].args[1]

        # verify that the stop log request is correct, with the first project name
        assert stop_log_request.project == "some-project-0"
        assert len(stop_log_request.runUIDs) == stop_logs_run_uids_chunk_size
        assert (
            deepdiff.DeepDiff(
                list(stop_log_request.runUIDs),
                # takes the first <stop_logs_run_uids_chunk_size> runs
                run_uids[:stop_logs_run_uids_chunk_size],
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
        log_collector = server.api.utils.clients.log_collector.LogCollectorClient()

        project_name = "some-project"

        # iterate over some runs, for each run assign different state
        run_uids_to_state = [
            ("some_uid", mlrun.runtimes.constants.RunStates.completed),
            ("some_uid2", mlrun.runtimes.constants.RunStates.unknown),
            ("some_uid3", mlrun.runtimes.constants.RunStates.completed),
            ("some_uid4", mlrun.runtimes.constants.RunStates.completed),
            # keep it last, as we later on omit it from the run_uids list
            ("some_uid5", mlrun.runtimes.constants.RunStates.running),
        ]
        for run_uid, state in run_uids_to_state:
            _create_new_run(
                db,
                project_name,
                uid=run_uid,
                name=run_uid,
                kind="job",
                state=state,
            )

        run_uids = [run_uid for run_uid, _ in run_uids_to_state]

        # the first run is not currently being log collected
        run_uids_log_collected = run_uids[1:]

        # update requested logs field to True
        server.api.utils.singletons.db.get_db().update_runs_requested_logs(
            db, run_uids, True
        )

        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[True],
            only_uids=False,
        )
        assert len(runs) == len(run_uids)

        log_collector._call = unittest.mock.AsyncMock(return_value=None)
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=ListRunsResponse(run_uids=run_uids_log_collected)
        )

        await server.api.main._verify_log_collection_stopped_on_startup()

        assert log_collector._call_stream.call_count == 1
        assert log_collector._call_stream.call_args[0][0] == "ListRunsInProgress"
        assert log_collector._call.call_count == 1
        assert log_collector._call.call_args[0][0] == "StopLogs"
        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name

        # one of the runs is in running state
        expected_run_uids = run_uids_log_collected[:-1]
        assert len(stop_log_request.runUIDs) == len(expected_run_uids)
        assert (
            deepdiff.DeepDiff(
                list(stop_log_request.runUIDs),
                expected_run_uids,
                ignore_order=True,
            )
            == {}
        )

        # update requested logs field to False for one run
        server.api.utils.singletons.db.get_db().update_runs_requested_logs(
            db, [run_uids[1]], False
        )

        runs = server.api.utils.singletons.db.get_db().list_distinct_runs_uids(
            db,
            requested_logs_modes=[True],
            only_uids=False,
        )
        assert len(runs) == 4

        # mock it again so the stream will run again
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=ListRunsResponse(run_uids=run_uids_log_collected)
        )

        await server.api.main._verify_log_collection_stopped_on_startup()

        assert log_collector._call.call_count == 2
        assert log_collector._call.call_args[0][0] == "StopLogs"
        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name
        assert len(stop_log_request.runUIDs) == 2

        # the first run is not currently being log collected, second run has requested logs set to False
        # and the last run is in running state
        expected_run_uids = run_uids_log_collected[1:-1]
        assert (
            deepdiff.DeepDiff(
                list(stop_log_request.runUIDs),
                expected_run_uids,
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
    function="",
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
    if function:
        run["spec"] = {"function": function}
    if store:
        server.api.crud.Runs().store_run(
            db_session, run, uid, iter=iteration, project=project
        )
    return project, name, uid, iteration, run
