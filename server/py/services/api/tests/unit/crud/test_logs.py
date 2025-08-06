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

import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors

import framework.utils.clients.log_collector
import services.api.crud
from services.api.tests.unit.utils.clients.test_log_collector import GetLogSizeResponse


class TestLogs:
    @staticmethod
    def test_legacy_log_mechanism(
        db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        project = "project-name"
        uid = "m33"
        data1, data2 = b"ab", b"cd"
        services.api.crud.Runs().store_run(
            db,
            {"metadata": {"name": "run-name"}, "some-run-data": "blabla"},
            uid,
            project=project,
        )
        services.api.crud.Logs().store_log(data1, project, uid)
        log = services.api.crud.Logs()._get_logs_legacy_method(db, project, uid)
        assert data1 == log, "get log 1"

        services.api.crud.Logs().store_log(data2, project, uid, append=True)
        log = services.api.crud.Logs()._get_logs_legacy_method(db, project, uid)
        assert data1 + data2 == log, "get log 2"

        services.api.crud.Logs().store_log(data1, project, uid, append=False)
        log = services.api.crud.Logs()._get_logs_legacy_method(db, project, uid)
        assert data1 == log, "get log append=False"

    @pytest.mark.parametrize(
        "return_value, expected_error",
        [
            (5, False),
            (0, False),
            (-1, True),
        ],
    )
    @pytest.mark.asyncio
    async def test_get_log_size(
        self, db: sqlalchemy.orm.Session, return_value, expected_error
    ):
        mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar
        log_collector = framework.utils.clients.log_collector.LogCollectorClient()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=GetLogSizeResponse(True, None, return_value)
        )

        project = "project-name"
        uid = "m33"

        services.api.crud.Runs().store_run(
            db,
            {"metadata": {"name": "run-name", "project": project, "uid": uid}},
            uid,
            project=project,
        )

        if expected_error:
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                await services.api.crud.Logs().get_log_size(db, project, uid)
        else:
            log_size = await services.api.crud.Logs().get_log_size(db, project, uid)
            assert return_value == log_size

    @pytest.mark.parametrize(
        "attempt, expected_run_uid",
        [
            (None, "m33-attempt-2"),
            (0, "m33-attempt-2"),
            (1, "m33"),
            (2, "m33-attempt-2"),
        ],
    )
    @pytest.mark.asyncio
    async def test_get_log_size_with_attempt(
        self, db: sqlalchemy.orm.Session, attempt, expected_run_uid
    ):
        mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar
        log_collector_client = (
            framework.utils.clients.log_collector.LogCollectorClient()
        )
        log_collector_client.get_log_size = unittest.mock.AsyncMock(return_value=5)

        project = "project-name"
        uid = "m33"

        services.api.crud.Runs().store_run(
            db,
            {
                "metadata": {"name": "run-name", "project": project, "uid": uid},
                "status": {"retry_count": 1},
            },
            uid,
            project=project,
        )

        log_size = await services.api.crud.Logs().get_log_size(
            db, project, uid, attempt=attempt
        )
        assert log_size == log_size
        log_collector_client.get_log_size.assert_called_once_with(
            project=project, run_uid=expected_run_uid
        )
