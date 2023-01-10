import mlrun.api.proto.log_collector_pb2
import mlrun.api.proto.log_collector_pb2_grpc

from .base import BaseGRPCClient


class LogCollectorClient(BaseGRPCClient):
    name = "log_collector"
    stub_class = mlrun.api.proto.log_collector_pb2_grpc.LogCollectorStub

    async def start_logs(self, run_id: str, selector: dict) -> bool:
        request = mlrun.api.proto.log_collector_pb2.StartLogRequest(runId=run_id, selector=selector)
        response = await self._call("StartLog", request)
        if not response.success:
            raise Exception(response.error)
        return response.success
