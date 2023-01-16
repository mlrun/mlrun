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

import mlrun.errors
import mlrun.utils.singleton
from mlrun.api.utils.clients.base_grpc import BaseGRPCClient
from mlrun.utils import logger


class LogCollectorClient(BaseGRPCClient, metaclass=mlrun.utils.singleton.Singleton):
    name = "log_collector"

    def __init__(self):
        super().__init__()
        self._initialize_proto_client_imports()
        self.stub_class = self._log_collector_pb2_grpc.LogCollectorStub

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
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> (bool, str):
        """
        Start logs streaming from the log collector service
        :param run_uid: The run uid
        :param selector: The selector to filter the logs by (e.g. "application=mlrun,job-name=job")
            format is key1=value1,key2=value2
        :param project: The project name
        :param verbose: Whether to log errors
        :param raise_on_error: Whether to raise an exception on error
        :return:
        """
        request = self._log_collector_pb2.StartLogRequest(
            runUID=run_uid, selector=selector, projectName=project
        )
        response = await self._call("StartLog", request)
        if not response.success:
            msg = f"Failed to start logs for run {run_uid}"
            if raise_on_error:
                raise mlrun.errors.MLRunInternalServerError(
                    msg,
                    error=response.error,
                )
            if verbose:
                logger.warning(msg, error=response.error)
        return response.success, response.error

    async def get_logs(
        self,
        run_uid: str,
        project: str,
        offset: int = 0,
        size: int = -1,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> bytes:
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
        request = self._log_collector_pb2.GetLogsRequest(
            runUID=run_uid,
            projectName=project,
            offset=offset,
            size=size,
        )
        response = await self._call("GetLogs", request)
        if not response.success:
            msg = f"Failed to get logs for run {run_uid}"
            if raise_on_error:
                raise mlrun.errors.MLRunInternalServerError(
                    msg,
                    error=response.error,
                )
            if verbose:
                logger.warning(msg, error=response.error)
        return response.logs
