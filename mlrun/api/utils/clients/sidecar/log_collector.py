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

from .base import BaseGRPCClient


class LogCollectorClient(BaseGRPCClient):
    name = "log_collector"
    stub_class = mlrun.api.proto.log_collector_pb2_grpc.LogCollectorStub

    async def start_logs(self, run_id: str, selector: str) -> bool:
        request = mlrun.api.proto.log_collector_pb2.StartLogRequest(
            runId=run_id, selector=selector
        )
        response = await self._call("StartLog", request)
        if not response.success:
            raise Exception(response.error)
        return response.success
