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

import google.protobuf.reflection
import grpc

import mlrun.api.schemas
import mlrun.config
import mlrun.errors


class BaseGRPCClient(object):

    name = None
    stub_class = None

    def __init__(self, address: str):
        self._address = address
        self._channel = None
        # A module acting as the interface for gRPC clients to call service methods.
        self._stub = None
        self._initialize()

    def is_initialized(self):
        return self._stub is not None

    def _initialize(self):
        if not self.name:
            return

        if not self._address:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "No address was provided, Unable to initialize client"
            )

        self._channel = grpc.aio.insecure_channel(self._address)
        if self.stub_class:
            self._stub = self.stub_class(self._channel)

    def _ensure_stub(self):
        if not self._stub:
            raise mlrun.errors.MLRunRuntimeError("Client not initialized")

    async def _call(
        self,
        endpoint: str,
        request: google.protobuf.reflection.GeneratedProtocolMessageType,
    ):
        """
        Call a unary endpoint of the gRPC server
        :param endpoint: The server endpoint to call
        :param request: The request to send
        :return: The server response
        """
        self._ensure_stub()
        response = await getattr(self._stub, endpoint)(request)
        return response

    def _call_stream(
        self,
        endpoint: str,
        request: google.protobuf.reflection.GeneratedProtocolMessageType,
    ):
        """
        Call a streaming endpoint of the gRPC server
        :param endpoint: The server endpoint to call
        :param request: The request to send to the server
        :return: A generator stream of the server responses
        """
        self._ensure_stub()
        response_stream = getattr(self._stub, endpoint)(request)
        return response_stream
