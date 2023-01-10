import grpc

import mlrun.config
from mlrun.utils import logger


class BaseGRPCClient(object):

    name = None
    stub_class = None

    def __init__(self):
        self._channel = None
        self._stub = None
        self._initialize()

    def _initialize(self):
        if not self.name:
            return
        sidecar_config = getattr(mlrun.config.config.sidecar, self.name)
        self._channel = grpc.aio.insecure_channel(sidecar_config.address)
        if self.stub_class:
            self._stub = self.stub_class(self._channel)

    async def _call(self, endpoint, request):
        logger.debug("Calling endpoint", endpoint=endpoint, request=request)
        response = await getattr(self._stub, endpoint)(request)
        return response
