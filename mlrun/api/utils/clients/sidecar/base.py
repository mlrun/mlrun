class GRPCCLient(object):
    def __init__(self, address, port, logger=None):
        self._address = address
        self._port = port
        self._logger = logger or get_logger("GRPCClient")
        self._channel = grpc.insecure_channel(f"{address}:{port}")
        self._client = grpc_client.GrpcClient(self._channel)
