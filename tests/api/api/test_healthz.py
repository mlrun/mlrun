# Copyright 2023 Iguazio
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
import http

import fastapi.testclient
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.config


def test_health(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    # sanity
    response = client.get("healthz")
    assert response.status_code == http.HTTPStatus.OK.value

    # fail
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.offline
    response = client.get("healthz")
    assert response.status_code == http.HTTPStatus.SERVICE_UNAVAILABLE.value
