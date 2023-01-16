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

import grpc

import mlrun.api.schemas
import mlrun.config
import mlrun.errors
from mlrun.utils import logger


class BaseGRPCClient(object):

    name = None
    stub_class = None

    def __init__(self):
        self._channel = None
        self._stub = None
        self._initialize()

    def is_initialized(self):
        return self._stub is not None

    def _initialize(self):
        if not self.name:
            return
        # get the config for the relevant client ( e.g "log_collector" ) meaning the name of the config needs to be
        # in the root of the config
        sidecar_config = getattr(mlrun.config.config, self.name)
        if sidecar_config.mode == mlrun.api.schemas.LogsCollectorMode.legacy:
            return
        self._channel = grpc.aio.insecure_channel(sidecar_config.address)
        if self.stub_class:
            self._stub = self.stub_class(self._channel)

    async def _call(self, endpoint, request):
        if not self._stub:
            raise mlrun.errors.MLRunRuntimeError("Client not initialized")
        logger.debug("Calling endpoint", endpoint=endpoint, request=request)
        response = await getattr(self._stub, endpoint)(request)
        return response
