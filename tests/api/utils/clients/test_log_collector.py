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

import unittest.mock

import deepdiff
import fastapi.testclient
import pytest
import sqlalchemy.orm.session

import mlrun
import mlrun.api.schemas
import mlrun.api.utils.clients.log_collector


class BaseLogCollectorResponse:
    def __init__(self, success, error):
        self.success = success
        self.errorMessage = error
        self.errorCode = (
            mlrun.api.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal
        )


class GetLogsResponse:
    def __init__(self, success, error, logs, total_calls):
        self.success = success
        self.errorMessage = error
        self.errorCode = (
            mlrun.api.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal
        )
        self.logs = logs
        self.total_calls = total_calls
        self.current_calls = 0

    # the following methods are required for the async iterator protocol
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current_calls < self.total_calls:
            self.current_calls += 1
            return self
        raise StopAsyncIteration


class HasLogsResponse:
    def __init__(self, success, error, has_logs):
        self.success = success
        self.errorMessage = error
        self.errorCode = (
            mlrun.api.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal
        )
        self.hasLogs = has_logs


mlrun.mlconf.log_collector.address = "http://localhost:8080"
mlrun.mlconf.log_collector.mode = mlrun.api.schemas.LogsCollectorMode.sidecar


class TestLogCollector:
    @pytest.mark.asyncio
    async def test_start_log(
        self,
        db: sqlalchemy.orm.session.Session,
        client: fastapi.testclient.TestClient,
        monkeypatch,
    ):
        run_uid = "123"
        project_name = "some-project"
        selector = f"mlrun/project={project_name},mlrun/uid={run_uid}"
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        success, error = await log_collector.start_logs(
            run_uid=run_uid, project=project_name, selector=selector
        )
        assert success is True and not error

        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "Failed to start logs")
        )
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.start_logs(
                run_uid=run_uid, project=project_name, selector=selector
            )

        success, error = await log_collector.start_logs(
            run_uid=run_uid,
            project=project_name,
            selector=selector,
            raise_on_error=False,
        )
        assert success is False and error == "Failed to start logs"

    @pytest.mark.asyncio
    async def test_get_logs(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        run_uid = "123"
        project_name = "some-project"
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        log_byte_string = b"some log"

        # mock responses for HasLogs and GetLogs
        log_collector._call = unittest.mock.AsyncMock(
            return_value=HasLogsResponse(True, "", True)
        )
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=GetLogsResponse(True, "", log_byte_string, 1)
        )

        log_stream = log_collector.get_logs(run_uid=run_uid, project=project_name)
        async for log in log_stream:
            assert log == log_byte_string

        # mock failed response for 5 calls for the next 2 tests, because get_logs retries 4 times
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=GetLogsResponse(False, "Failed to get logs", b"", 5),
        )
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            async for log in log_collector.get_logs(
                run_uid=run_uid, project=project_name
            ):
                assert log == b""  # should not get here

        # mock HasLogs response to return False
        log_collector._call = unittest.mock.AsyncMock(
            return_value=HasLogsResponse(True, "", False)
        )

        log_stream = log_collector.get_logs(
            run_uid=run_uid, project=project_name, raise_on_error=False
        )
        async for log in log_stream:
            assert log == b""

    @pytest.mark.asyncio
    async def test_get_log_with_retryable_error(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        run_uid = "123"
        project_name = "some-project"
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        # mock responses for HasLogs to return a retryable error
        log_collector._call = unittest.mock.AsyncMock(
            return_value=HasLogsResponse(
                False,
                "readdirent /var/mlrun/logs/blabla: resource temporarily unavailable",
                True,
            )
        )

        log_stream = log_collector.get_logs(
            run_uid=run_uid, project=project_name, raise_on_error=False
        )
        async for log in log_stream:
            assert log == b""

        # mock responses for HasLogs to return a retryable error
        log_collector._call = unittest.mock.AsyncMock(
            return_value=HasLogsResponse(
                False,
                "I'm an error that should not be retried",
                True,
            )
        )
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            async for log in log_collector.get_logs(
                run_uid=run_uid, project=project_name
            ):
                assert log == b""  # should not get here

    @pytest.mark.asyncio
    async def test_stop_logs(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        run_uids = ["123"]
        project_name = "some-project"
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        # test successful stop logs
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        await log_collector.stop_logs(run_uids=run_uids, project=project_name)
        assert log_collector._call.call_count == 1
        assert log_collector._call.call_args[0][0] == "StopLogs"

        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name
        assert stop_log_request.runUIDs == run_uids

        # test failed stop logs
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "Failed to stop logs")
        )
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.stop_logs(run_uids=run_uids, project=project_name)

    @pytest.mark.asyncio
    async def test_delete_logs(
        self, db: sqlalchemy.orm.session.Session, client: fastapi.testclient.TestClient
    ):
        run_uids = None
        project_name = "some-project"
        log_collector = mlrun.api.utils.clients.log_collector.LogCollectorClient()

        # test successful stop logs
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(True, "")
        )
        await log_collector.delete_logs(run_uids=run_uids, project=project_name)
        assert log_collector._call.call_count == 1
        assert log_collector._call.call_args[0][0] == "DeleteLogs"

        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name
        assert stop_log_request.runUIDs == []

        # test failed stop logs
        run_uids = ["123"]
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "Failed to delete logs")
        )
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.delete_logs(run_uids=run_uids, project=project_name)

        assert log_collector._call.call_count == 1
        assert log_collector._call.call_args[0][0] == "DeleteLogs"

        stop_log_request = log_collector._call.call_args[0][1]
        assert stop_log_request.project == project_name
        assert stop_log_request.runUIDs == run_uids

    @pytest.mark.parametrize(
        "error_code,expected_mlrun_error",
        [
            (0, mlrun.errors.MLRunNotFoundError),
            (1, mlrun.errors.MLRunInternalServerError),
            (2, mlrun.errors.MLRunBadRequestError),
        ],
    )
    def test_log_collector_error_mapping(self, error_code, expected_mlrun_error):
        failure_message = "some failure message"
        error_message = "some error message"
        error = mlrun.api.utils.clients.log_collector.LogCollectorErrorCode.map_error_code_to_mlrun_error(
            error_code, error_message, failure_message
        )

        message = f"{failure_message}, error: {error_message}"
        assert (
            deepdiff.DeepDiff(
                error,
                expected_mlrun_error(message),
            )
            == {}
        )
