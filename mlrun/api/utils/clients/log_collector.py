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
import enum
import http
import re
import typing

import mlrun.api.utils.clients.protocols.grpc
import mlrun.errors
import mlrun.utils.singleton
from mlrun.config import config
from mlrun.utils import logger


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
    def has_logs_retryable_errors(cls):
        return [
            cls.readdirent_resource_temporarily_unavailable,
        ]


class LogCollectorClient(
    mlrun.api.utils.clients.protocols.grpc.BaseGRPCClient,
    metaclass=mlrun.utils.singleton.Singleton,
):
    name = "log_collector"

    def __init__(self, address: str = None):
        self._initialize_proto_client_imports()
        self.stub_class = self._log_collector_pb2_grpc.LogCollectorStub
        super().__init__(address=address or mlrun.mlconf.log_collector.address)

    def _initialize_proto_client_imports(self):
        # Importing the proto client classes here and not at the top of the file to avoid raising an import error
        # when the log_collector service is not enabled / the proto client wasn't compiled
        import mlrun.api.proto.log_collector_pb2
        import mlrun.api.proto.log_collector_pb2_grpc

        self._log_collector_pb2 = mlrun.api.proto.log_collector_pb2
        self._log_collector_pb2_grpc = mlrun.api.proto.log_collector_pb2_grpc

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
            has_logs = await self.has_logs(run_uid, project, verbose, raise_on_error)
            if not has_logs:
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
        while True:
            try:
                response_stream = self._call_stream("GetLogs", request)
                async for chunk in response_stream:
                    if not chunk.success:
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
                    raise mlrun.errors.raise_for_status_code(
                        http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                        mlrun.errors.err_to_str(exc),
                    )

                # breath
                await asyncio.sleep(3)

    async def has_logs(
        self,
        run_uid: str,
        project: str,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> bool:
        """
        Check if the log collector service has logs for the given run
        :param run_uid: The run uid
        :param project: The project name
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return: Whether the log collector service has logs for the given run
        """
        request = self._log_collector_pb2.HasLogsRequest(
            runUID=run_uid, projectName=project
        )

        response = await self._call("HasLogs", request)
        if not response.success:
            if self._retryable_error(
                response.errorMessage,
                LogCollectorErrorRegex.has_logs_retryable_errors(),
            ):
                if verbose:
                    logger.warning(
                        "Failed to check if run has logs to collect, retrying",
                        run_uid=run_uid,
                        error=response.errorMessage,
                    )
                return False

            msg = f"Failed to check if run has logs to collect for {run_uid}"
            if verbose:
                logger.warning(msg, error=response.errorMessage)
            if raise_on_error:
                raise LogCollectorErrorCode.map_error_code_to_mlrun_error(
                    response.errorCode, response.errorMessage, msg
                )
        return response.hasLogs

    async def stop_logs(
        self,
        project: str,
        run_uids: typing.List[str] = None,
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
        run_uids: typing.List[str] = None,
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

    def _retryable_error(self, error_message, retryable_error_patterns) -> bool:
        """
        Check if the error is retryable
        :param error_message: The error message
        :param retryable_error_patterns: The retryable error regex patterns
        :return: Whether the error is retryable
        """
        if any(re.match(regex, error_message) for regex in retryable_error_patterns):
            return True
        return False
