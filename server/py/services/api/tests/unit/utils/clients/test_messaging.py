# Copyright 2024 Iguazio
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
import unittest.mock

import fastapi
import fastapi.testclient

from mlrun.utils import logger

import framework.utils.clients.discovery
import framework.utils.clients.messaging

test_router = fastapi.APIRouter()


@test_router.get("/success")
def success(request: fastapi.Request):
    logger.info("Success endpoint received request, returning 202")
    return fastapi.Response(status_code=202)


def test_messaging_client_forward_request(client: fastapi.testclient.TestClient):
    messaging_client = framework.utils.clients.messaging.Client()
    messaging_client._discovery = unittest.mock.Mock(
        return_value=framework.utils.clients.discovery.ServiceInstance(
            name="test", url="bla"
        )
    )
