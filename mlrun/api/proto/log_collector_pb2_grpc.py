# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import log_collector_pb2 as proto_dot_log__collector__pb2


class LogCollectorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.StartLog = channel.unary_unary(
            "/mlrun.LogCollector/StartLog",
            request_serializer=proto_dot_log__collector__pb2.StartLogRequest.SerializeToString,
            response_deserializer=proto_dot_log__collector__pb2.StartLogResponse.FromString,
        )
        self.GetLogs = channel.unary_unary(
            "/mlrun.LogCollector/GetLogs",
            request_serializer=proto_dot_log__collector__pb2.GetLogsRequest.SerializeToString,
            response_deserializer=proto_dot_log__collector__pb2.GetLogsResponse.FromString,
        )


class LogCollectorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def StartLog(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetLogs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_LogCollectorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "StartLog": grpc.unary_unary_rpc_method_handler(
            servicer.StartLog,
            request_deserializer=proto_dot_log__collector__pb2.StartLogRequest.FromString,
            response_serializer=proto_dot_log__collector__pb2.StartLogResponse.SerializeToString,
        ),
        "GetLogs": grpc.unary_unary_rpc_method_handler(
            servicer.GetLogs,
            request_deserializer=proto_dot_log__collector__pb2.GetLogsRequest.FromString,
            response_serializer=proto_dot_log__collector__pb2.GetLogsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "mlrun.LogCollector", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class LogCollector(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def StartLog(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/mlrun.LogCollector/StartLog",
            proto_dot_log__collector__pb2.StartLogRequest.SerializeToString,
            proto_dot_log__collector__pb2.StartLogResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def GetLogs(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/mlrun.LogCollector/GetLogs",
            proto_dot_log__collector__pb2.GetLogsRequest.SerializeToString,
            proto_dot_log__collector__pb2.GetLogsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
