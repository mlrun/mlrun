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

import mlrun.api.proto.log_collector_pb2
import mlrun.api.proto.log_collector_pb2_grpc
import mlrun.errors
from mlrun.utils import logger

from .base import BaseGRPCClient


class LogCollectorClient(BaseGRPCClient):
    name = "log_collector"
    stub_class = mlrun.api.proto.log_collector_pb2_grpc.LogCollectorStub

    async def start_logs(
        self,
        run_id: str,
        selector: str,
        verbose: bool = True,
        raise_on_error: bool = True,
    ) -> (bool, str):
        request = mlrun.api.proto.log_collector_pb2.StartLogRequest(
            runId=run_id, selector=selector
        )
        response = await self._call("StartLog", request)
        if not response.success:
            msg = f"Failed to start logs for run {run_id}"
            if raise_on_error:
                raise mlrun.errors.MLRunInternalServerError(
                    msg,
                    error=response.error,
                )
            if verbose:
                logger.warning(msg, error=response.error)
        return response.success, response.error
