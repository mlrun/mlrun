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

import deepdiff
import pytest

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas

import framework.utils.clients.log_collector


class BaseLogCollectorResponse:
    def __init__(self, success, error):
        self.success = success
        self.errorMessage = error
        self.errorCode = (
            framework.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal
        )


class GetLogsResponse:
    def __init__(self, success, error, logs, total_calls):
        self.success = success
        self.errorMessage = error
        self.errorCode = (
            framework.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal
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


class GetLogSizeResponse:
    def __init__(self, success, error, log_size=None):
        self.success = success
        self.errorMessage = error
        self.errorCode = (
            framework.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal
        )
        self.logSize = log_size


class ListRunsResponse:
    def __init__(self, run_uids=None, total_calls=1):
        self.runUIDs = run_uids or []
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


mlrun.mlconf.log_collector.address = "http://localhost:8080"
mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar


class TestLogCollector:
    @pytest.mark.asyncio
    async def test_start_log(
        self,
        monkeypatch,
    ):
        run_uid = "123"
        project_name = "some-project"
        selector = (
            f"{mlrun_constants.MLRunInternalLabels.project}={project_name},"
            f"{mlrun_constants.MLRunInternalLabels.uid}={run_uid}"
        )
        log_collector = self._client_with_listener()

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
    async def test_get_logs(self):
        run_uid = "123"
        project_name = "some-project"
        log_collector = self._client_with_listener()

        log_byte_string = b"some log"

        # mock responses for GetLogSize and GetLogs
        log_collector._call = unittest.mock.AsyncMock(
            return_value=GetLogSizeResponse(True, "", 1)
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

        # mock GetLogSize response to return 0
        log_collector._call = unittest.mock.AsyncMock(
            return_value=GetLogSizeResponse(True, "", 0)
        )

        log_stream = log_collector.get_logs(
            run_uid=run_uid, project=project_name, raise_on_error=False
        )
        async for log in log_stream:
            assert log == b""

    @pytest.mark.asyncio
    async def test_get_log_with_retryable_error(self):
        run_uid = "123"
        project_name = "some-project"
        log_collector = self._client_with_listener()

        # mock responses for GetLogSize to return a retryable error
        log_collector._call = unittest.mock.AsyncMock(
            return_value=GetLogSizeResponse(
                False,
                "readdirent /var/mlrun/logs/blabla: resource temporarily unavailable",
            )
        )

        log_stream = log_collector.get_logs(
            run_uid=run_uid, project=project_name, raise_on_error=False
        )
        async for log in log_stream:
            assert log == b""

        # mock responses for GetLogSize to return a non-retryable error
        log_collector._call = unittest.mock.AsyncMock(
            return_value=GetLogSizeResponse(
                False,
                "I'm an error that should not be retried",
            )
        )
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            async for log in log_collector.get_logs(
                run_uid=run_uid, project=project_name
            ):
                assert log == b""  # should not get here

    @pytest.mark.asyncio
    async def test_stop_logs(self):
        run_uids = ["123"]
        project_name = "some-project"
        log_collector = self._client_with_listener()

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
    async def test_delete_logs(self):
        run_uids = None
        project_name = "some-project"
        log_collector = self._client_with_listener()

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

    @pytest.mark.asyncio
    async def test_list_runs_in_progress(self):
        project_name = "some-project"
        log_collector = self._client_with_listener()

        async def _verify_runs(run_uids_stream):
            async for run_uid_list in run_uids_stream:
                for run_uid in run_uid_list:
                    assert run_uid in run_uids

        # mock a short response for ListRunsInProgress
        run_uids = [f"{str(i)}" for i in range(10)]
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=ListRunsResponse(run_uids=run_uids)
        )
        run_uids_stream = log_collector.list_runs_in_progress(project=project_name)
        await _verify_runs(run_uids_stream)

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
        error = framework.utils.clients.log_collector.LogCollectorErrorCode.map_error_code_to_mlrun_error(
            error_code, error_message, failure_message
        )

        message = f"{failure_message}, error: {error_message}"
        assert (
            deepdiff.DeepDiff(
                str(error),
                str(expected_mlrun_error(message)),
            )
            == {}
        )


class _NotifiableResponse:
    """Response shape with an int ``errorCode`` matching what gRPC emits in
    production (the proto field is int32). Tests asserting NotFound vs. other
    error-code paths need real ints to drive the production filter correctly."""

    def __init__(self, success, error, error_code, logs=None, total_calls=0):
        self.success = success
        self.errorMessage = error
        self.errorCode = error_code
        if logs is not None:
            self.logs = logs
        if total_calls:
            self.total_calls = total_calls
            self.current_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current_calls < self.total_calls:
            self.current_calls += 1
            return self
        raise StopAsyncIteration


_NOT_FOUND_CODE = (
    framework.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeNotFound.value
)
_INTERNAL_CODE = (
    framework.utils.clients.log_collector.LogCollectorErrorCode.ErrCodeInternal.value
)


class TestLogCollectorFailureListener:
    """Coverage for the failure-listener hook on the framework log-collector
    client. The listener is registered by ``services.api.utils.events.log_collector_errors``
    in production; here we register a no-op capture so we can assert which RPC
    failure paths call it and which intentionally skip it.

    The gRPC stub init in ``LogCollectorClient.__init__`` imports a generated
    proto module that requires ``make schemas`` to have been run. These tests
    don't exercise gRPC at all (``_call`` / ``_call_stream`` are mocked), so we
    short-circuit the proto import to keep the suite runnable in environments
    where protos haven't been generated."""

    @pytest.fixture(autouse=True)
    def _reset_listeners(self, monkeypatch):
        # The real __init__ imports a generated grpc stub module that requires
        # `make schemas`; for these tests we only need a LogCollectorClient
        # whose `_call` / `_call_stream` we mock, so stub the proto wiring.
        # Singleton instances themselves are wiped between tests by the autouse
        # `config_test_base` fixture in tests/common_fixtures.py.
        def _stub_proto_init(self):
            self._log_collector_pb2 = unittest.mock.MagicMock()
            self._log_collector_pb2_grpc = unittest.mock.MagicMock()

        monkeypatch.setattr(
            framework.utils.clients.log_collector.LogCollectorClient,
            "_initialize_proto_client_imports",
            _stub_proto_init,
        )
        self.calls: list[dict] = []
        # NB: construct LogCollectorClient inside each test, not here — the
        # gRPC base client needs a running event loop and this fixture runs
        # outside one.

    def _client_with_listener(self):
        """Singleton LogCollectorClient with the capture listener attached."""
        client = framework.utils.clients.log_collector.LogCollectorClient()
        client.add_failure_listener(self.calls.append)
        return client

    @pytest.mark.asyncio
    async def test_start_logs_failure_notifies(self):
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=_NotifiableResponse(
                success=False, error="collector down", error_code=_INTERNAL_CODE
            )
        )

        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.start_logs(
                run_uid="r1", selector="application=mlrun", project="p1"
            )

        assert len(self.calls) == 1
        ctx = self.calls[0]
        assert ctx["operation"] == "start_logs"
        assert ctx["run_uid"] == "r1"
        assert ctx["project"] == "p1"
        assert ctx["error_category"] == "start_logs_failed"
        assert ctx["error_code"] == _INTERNAL_CODE

    @pytest.mark.asyncio
    async def test_start_logs_not_found_does_not_notify(self):
        """ErrCodeNotFound = "logs not yet produced" — benign, must not fire
        a MAJOR system event."""
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=_NotifiableResponse(
                success=False, error="run not found", error_code=_NOT_FOUND_CODE
            )
        )

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            await log_collector.start_logs(
                run_uid="r1", selector="application=mlrun", project="p1"
            )

        assert self.calls == []

    @pytest.mark.asyncio
    async def test_get_logs_chunk_failure_notifies_once(self):
        """A failed multi-chunk stream notifies on the first failing chunk
        only — subsequent failing chunks must not re-notify (per-call
        deduplication, distinct from the 60s process-level throttle)."""
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=_NotifiableResponse(
                success=True, error="", error_code=_INTERNAL_CODE
            )
        )
        # GetLogSize succeeds with size>0, then GetLogs stream yields 3 failing chunks.
        size_response = GetLogSizeResponse(True, "", 1)
        get_logs_stream = _NotifiableResponse(
            success=False,
            error="stream broken",
            error_code=_INTERNAL_CODE,
            logs=b"",
            total_calls=3,
        )
        log_collector._call = unittest.mock.AsyncMock(return_value=size_response)
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=get_logs_stream
        )

        async for _ in log_collector.get_logs(
            run_uid="r1", project="p1", raise_on_error=False
        ):
            pass

        assert len(self.calls) == 1
        assert self.calls[0]["operation"] == "get_logs"
        assert self.calls[0]["error_category"] == "get_logs_failed"

    @pytest.mark.asyncio
    async def test_get_logs_not_found_chunk_does_not_notify(self):
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=GetLogSizeResponse(True, "", 1)
        )
        log_collector._call_stream = unittest.mock.MagicMock(
            return_value=_NotifiableResponse(
                success=False,
                error="logs not yet collected",
                error_code=_NOT_FOUND_CODE,
                logs=b"",
                total_calls=1,
            )
        )

        async for _ in log_collector.get_logs(
            run_uid="r1", project="p1", raise_on_error=False
        ):
            pass

        assert self.calls == []

    @pytest.mark.asyncio
    async def test_get_log_size_failure_notifies(self):
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=_NotifiableResponse(
                success=False, error="kaboom", error_code=_INTERNAL_CODE
            )
        )

        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.get_log_size(run_uid="r1", project="p1")

        assert len(self.calls) == 1
        assert self.calls[0]["operation"] == "get_log_size"
        assert self.calls[0]["error_category"] == "get_log_size_failed"

    @pytest.mark.asyncio
    async def test_get_log_size_readdirent_retryable_does_not_notify(self):
        """The readdirent transient-error path returns 0 without raising; it
        is self-healing and must not fire a MAJOR event."""
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=_NotifiableResponse(
                success=False,
                error="readdirent /var/mlrun/logs/proj: resource temporarily unavailable",
                error_code=_INTERNAL_CODE,
            )
        )

        size = await log_collector.get_log_size(run_uid="r1", project="p1")
        assert size == 0
        assert self.calls == []

    @pytest.mark.asyncio
    async def test_stop_logs_failure_does_not_notify(self):
        """Lifecycle RPCs are out of scope per the spec wording 'failed to
        retrieve logs'."""
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "stop failed")
        )

        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.stop_logs(project="p1", run_uids=["r1"])

        assert self.calls == []

    @pytest.mark.asyncio
    async def test_delete_logs_failure_does_not_notify(self):
        log_collector = self._client_with_listener()
        log_collector._call = unittest.mock.AsyncMock(
            return_value=BaseLogCollectorResponse(False, "delete failed")
        )

        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await log_collector.delete_logs(project="p1", run_uids=["r1"])

        assert self.calls == []

    @pytest.mark.asyncio
    async def test_failure_listener_raise_is_swallowed(self):
        """A misbehaving listener must not break the underlying RPC."""

        def bad_listener(_ctx):
            raise RuntimeError("listener bug")

        client = framework.utils.clients.log_collector.LogCollectorClient()
        client.add_failure_listener(bad_listener)
        client._call = unittest.mock.AsyncMock(
            return_value=_NotifiableResponse(
                success=False, error="boom", error_code=_INTERNAL_CODE
            )
        )

        # The RPC's own failure mode (raise) must still surface; the listener
        # exception is swallowed.
        with pytest.raises(mlrun.errors.MLRunInternalServerError):
            await client.start_logs(
                run_uid="r1", selector="application=mlrun", project="p1"
            )

    @pytest.mark.asyncio
    async def test_add_failure_listener_is_idempotent(self):
        # Async to ensure we're inside an event loop when LogCollectorClient
        # is constructed (the gRPC base needs a running loop).
        def listener(_ctx):
            pass

        client = framework.utils.clients.log_collector.LogCollectorClient()
        client.add_failure_listener(listener)
        client.add_failure_listener(listener)
        assert client._failure_listeners.count(listener) == 1
