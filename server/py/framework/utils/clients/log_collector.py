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
import dataclasses
import enum
import http
import re
import typing

import mlrun.errors
import mlrun.utils.singleton
from mlrun.config import config
from mlrun.utils import logger

import framework.utils.clients.protocols.grpc


@dataclasses.dataclass(frozen=True)
class LogCollectorFailureContext:
    """
    Payload delivered to failure listeners registered on
    :class:`LogCollectorClient`. Distinct values are passed for the operation
    that failed and the scope it was invoked with; ``error`` and ``error_code``
    are best-effort and may be absent on transport-level failures.
    """

    operation: str
    error_category: str
    run_uid: str | None = None
    project: str | None = None
    error: BaseException | str | None = None
    error_code: int | str | None = None


LogCollectorFailureListener = typing.Callable[[LogCollectorFailureContext], None]


class LogCollectorErrorCode(enum.Enum):
    ErrCodeNotFound = 0
    ErrCodeInternal = 1
    ErrCodeBadRequest = 2

    @staticmethod
    def map_error_code_to_mlrun_error(
        error_code: int, error_message: str, failure_message: str
    ) -> mlrun.errors.MLRunHTTPStatusError:
        """
        Map error code to exception
        :param error_code: The error code
        :param error_message: The error message
        :param failure_message: The failure message to use in the exception, according to the failed request
        """
        message = f"{failure_message}, error: {error_message}"

        mlrun_error_class = {
            LogCollectorErrorCode.ErrCodeNotFound: mlrun.errors.MLRunNotFoundError,
            LogCollectorErrorCode.ErrCodeInternal: mlrun.errors.MLRunInternalServerError,
            LogCollectorErrorCode.ErrCodeBadRequest: mlrun.errors.MLRunBadRequestError,
        }.get(
            LogCollectorErrorCode(error_code),
            mlrun.errors.MLRunInternalServerError,
        )

        return mlrun_error_class(message)


class LogCollectorErrorRegex:
    # when multiple routines in the log collector service try to search the same directory,
    # one of them can fail with this error
    readdirent_resource_temporarily_unavailable = (
        "readdirent.*resource temporarily unavailable"
    )

    @classmethod
    def get_log_size_retryable_errors(cls):
        return [
            cls.readdirent_resource_temporarily_unavailable,
        ]


class LogCollectorClient(
    framework.utils.clients.protocols.grpc.BaseGRPCClient,
    metaclass=mlrun.utils.singleton.Singleton,
):
    """
    gRPC client for the log-collector sidecar.

    A failure listener (see :meth:`set_failure_listener`) is invoked when a
    log-retrieval RPC (``start_logs``, ``get_logs``, ``get_log_size``)
    hard-fails. Lifecycle and inventory RPCs (``stop_logs``, ``delete_logs``,
    ``list_runs_in_progress``) intentionally do not notify — they are not
    "failed to retrieve logs" per the event spec. This hook is the only
    inbound coupling from telemetry: keeping it out of the data path lets
    ``services.api.utils.events`` register a publisher without the framework
    layer depending on services-layer modules.
    """

    name = "log_collector"

    def __init__(self, address: str | None = None):
        self._initialize_proto_client_imports()
        self.stub_class = self._log_collector_pb2_grpc.LogCollectorStub
        self._failure_listener: LogCollectorFailureListener | None = None
        super().__init__(address=address or mlrun.mlconf.log_collector.address)

    def set_failure_listener(self, listener: LogCollectorFailureListener) -> None:
        """
        Install the callback to be notified when a log-retrieval RPC fails.
        The callback receives a :class:`LogCollectorFailureContext` describing
        the failed operation, its scope, and (best-effort) the underlying
        error. Calling again replaces the previously installed listener.

        :param listener: callable invoked synchronously after each
            retrieval-RPC failure; must not raise. Exceptions are caught and
            logged.
        """
        self._failure_listener = listener

    def _notify_failure(self, context: LogCollectorFailureContext) -> None:
        """Invoke the registered failure listener; never raise."""
        listener = self._failure_listener
        if listener is None:
            return
        try:
            listener(context)
        except Exception as exc:
            logger.warning(
                "Log collector failure listener raised, ignoring",
                exc=mlrun.errors.err_to_str(exc),
            )

    def _initialize_proto_client_imports(self):
        # Importing the proto client classes here and not at the top of the file to avoid raising an import error
        # when the log_collector service is not enabled / the proto client wasn't compiled
        import schemas.proto.log_collector_pb2
        import schemas.proto.log_collector_pb2_grpc

        self._log_collector_pb2 = schemas.proto.log_collector_pb2
        self._log_collector_pb2_grpc = schemas.proto.log_collector_pb2_grpc

    async def start_logs(
        self,
        run_uid: str,
        selector: str,
        project: str = "",
        best_effort: bool = False,
        verbose: bool = False,
        raise_on_error: bool = True,
    ) -> (bool, str):
        """
        Start logs streaming from the log collector service
        :param run_uid: The run uid
        :param selector: The selector to filter the logs by (e.g. "application=mlrun,job-name=job")
            format is key1=value1,key2=value2
        :param project: The project name
        :param best_effort: Whether to start logs collection in best-effort mode, meaning that success will be returned
            even if the logs collection failed to start (e.g. if the pod doesn't exist)
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: A tuple of (success, error)
        """
        request = self._log_collector_pb2.StartLogRequest(
            runUID=run_uid,
            selector=selector,
            projectName=project,
            bestEffort=best_effort,
        )
        logger.debug(
            "Starting logs", run_uid=run_uid, selector=selector, project=project
        )
        response = await self._call("StartLog", request)
        if not response.success:
            if response.errorCode != LogCollectorErrorCode.ErrCodeNotFound.value:
                self._notify_failure(
                    LogCollectorFailureContext(
                        operation="start_logs",
                        error_category="start_logs_failed",
                        run_uid=run_uid,
                        project=project,
                        error=response.errorMessage,
                        error_code=response.errorCode,
                    )
                )
            msg = f"Failed to start logs for run {run_uid}"
            if raise_on_error:
                raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                    response.errorCode, response.errorMessage, msg
                )
            if verbose:
                logger.warning(msg, error=response.errorMessage)
        return response.success, response.errorMessage

    async def get_logs(
        self,
        run_uid: str,
        project: str,
        offset: int = 0,
        size: int = -1,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> typing.AsyncIterable[bytes]:
        """
        Get logs from the log collector service
        :param run_uid: The run uid
        :param project: The project name
        :param offset: The offset to start reading from
        :param size: The number of bytes to read (-1 for all)
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: The logs bytes
        """

        # check if this run has logs to collect
        try:
            log_size = await self.get_log_size(
                run_uid, project, verbose, raise_on_error
            )
            if log_size <= 0:
                logger.debug(
                    "Run has no logs to collect",
                    run_uid=run_uid,
                    project=project,
                )

                # run has no logs - return empty logs and exit so caller won't wait for logs or retry
                yield b""
                return
        except mlrun.errors.MLRunInternalServerError as exc:
            logger.warning(
                "Failed to check if run has logs to collect", run_uid=run_uid
            )
            if raise_on_error:
                raise mlrun.errors.MLRunInternalServerError(
                    f"Failed to check if run has logs to collect for {run_uid}. exception= {exc}"
                )

        request = self._log_collector_pb2.GetLogsRequest(
            runUID=run_uid,
            projectName=project,
            offset=offset,
            size=size,
        )

        # retry calling the server, it can fail in case the log-collector hasn't started collecting logs for this yet
        # TODO: add async retry function
        try_count = 0
        notified = False
        while True:
            try:
                response_stream = self._call_stream("GetLogs", request)
                async for chunk in response_stream:
                    if not chunk.success:
                        if (
                            not notified
                            and chunk.errorCode
                            != LogCollectorErrorCode.ErrCodeNotFound.value
                        ):
                            self._notify_failure(
                                LogCollectorFailureContext(
                                    operation="get_logs",
                                    error_category="get_logs_failed",
                                    run_uid=run_uid,
                                    project=project,
                                    error=chunk.errorMessage,
                                    error_code=chunk.errorCode,
                                )
                            )
                            notified = True
                        msg = f"Failed to get logs for run {run_uid}"
                        if raise_on_error:
                            raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                                chunk.errorCode, chunk.errorMessage, msg
                            )
                        if verbose:
                            logger.warning(msg, error=chunk.errorMessage)
                    yield chunk.logs
                return
            except Exception as exc:
                try_count += 1
                logger.warning(
                    "Failed to get logs, retrying",
                    try_count=try_count,
                    exc=mlrun.errors.err_to_str(exc),
                )
                if try_count == config.log_collector.get_logs.max_retries:
                    if not notified:
                        self._notify_failure(
                            LogCollectorFailureContext(
                                operation="get_logs",
                                error_category="get_logs_failed",
                                run_uid=run_uid,
                                project=project,
                                error=exc,
                            )
                        )
                        notified = True
                    raise mlrun.errors.err_for_status_code(
                        http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                        mlrun.errors.err_to_str(exc),
                    )

                # breath
                await asyncio.sleep(3)

    async def get_log_size(
        self,
        run_uid: str,
        project: str,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> int:
        """
        Returns the log file size for the given run
        :param run_uid: The run uid
        :param project: The project name
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: The log file size of the run, if it exists
        """
        request = self._log_collector_pb2.GetLogSizeRequest(
            runUID=run_uid, projectName=project
        )

        response = await self._call("GetLogSize", request)
        if not response.success:
            if self._retryable_error(
                response.errorMessage,
                LogCollectorErrorRegex.get_log_size_retryable_errors(),
            ):
                if verbose:
                    logger.warning(
                        "Failed to get log file size, retrying",
                        run_uid=run_uid,
                        error=response.errorMessage,
                    )
                return 0

            if response.errorCode != LogCollectorErrorCode.ErrCodeNotFound.value:
                self._notify_failure(
                    LogCollectorFailureContext(
                        operation="get_log_size",
                        error_category="get_log_size_failed",
                        run_uid=run_uid,
                        project=project,
                        error=response.errorMessage,
                        error_code=response.errorCode,
                    )
                )
            msg = f"Failed to log file size for {run_uid}"
            if verbose:
                logger.warning(msg, error=response.errorMessage)
            if raise_on_error:
                raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                    response.errorCode, response.errorMessage, msg
                )
        return response.logSize

    async def stop_logs(
        self,
        project: str,
        run_uids: list[str] | None = None,
        verbose: bool = False,
        raise_on_error: bool = True,
    ) -> None:
        """
        Stop logs streaming from the log collector service
        :param project: The project name
        :param run_uids: The run uids to stop logs for, if not provided will stop logs for all runs in the project
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: None
        """
        request = self._log_collector_pb2.StopLogsRequest(
            project=project, runUIDs=run_uids
        )

        response = await self._call("StopLogs", request)
        if not response.success:
            msg = "Failed to stop logs"
            if raise_on_error:
                raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                    response.errorCode, response.errorMessage, msg
                )
            if verbose:
                logger.warning(msg, error=response.errorMessage)

    async def delete_logs(
        self,
        project: str,
        run_uids: list[str] | None = None,
        verbose: bool = False,
        raise_on_error: bool = True,
    ) -> None:
        """
        Delete logs from the log collector service
        :param project: The project name
        :param run_uids: The run uids to delete logs for, if not provided will delete logs for all runs in the project
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: None
        """

        request = self._log_collector_pb2.StopLogsRequest(
            project=project, runUIDs=run_uids
        )

        response = await self._call("DeleteLogs", request)
        if not response.success:
            msg = "Failed to delete logs"
            if raise_on_error:
                raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                    response.errorCode, response.errorMessage, msg
                )
            if verbose:
                logger.warning(msg, error=response.errorMessage)

    async def list_runs_in_progress(
        self,
        project: str | None = None,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> typing.AsyncIterable[str]:
        """
        List runs in progress from the log collector service
        :param project: A project name to filter the runs by. If not provided, all runs in progress will be listed
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: A list of run uids
        """
        request = self._log_collector_pb2.ListRunsRequest(
            project=project,
        )

        response_stream = self._call_stream("ListRunsInProgress", request)
        try:
            async for chunk in response_stream:
                yield chunk.runUIDs
        except Exception as exc:
            msg = "Failed to list runs in progress"
            if raise_on_error:
                raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                    LogCollectorErrorCode.ErrCodeInternal.value,
                    mlrun.errors.err_to_str(exc),
                    msg,
                )
            if verbose:
                logger.warning(msg, error=mlrun.errors.err_to_str(exc))

    @staticmethod
    def _retryable_error(error_message, retryable_error_patterns) -> bool:
        """
        Check if the error is retryable
        :param error_message: The error message
        :param retryable_error_patterns: The retryable error regex patterns
        :return: Whether the error is retryable
        """
        if any(re.match(regex, error_message) for regex in retryable_error_patterns):
            return True
        return False
