# Copyright 2025 Iguazio
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
#
from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
from mlrun.common.types import AuthenticationMode

API_USER_SECRETS_PATH = "/user-secrets"
API_USER_SECRETS_TOKENS_PATH = API_USER_SECRETS_PATH + "/tokens"


def test_iguazio_v4_only_dependency(db: Session, client: TestClient):
    # Force unsupported auth mode
    mlrun.mlconf.httpdb.authentication.mode = AuthenticationMode.BASIC

    # Pick an endpoint that includes the iguazio_v4_only dependency
    response = client.put(API_USER_SECRETS_TOKENS_PATH, json=[])

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
