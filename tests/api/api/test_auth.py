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
#
import http

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.auth.verifier


def test_verify_authorization(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    authorization_verification_input = mlrun.api.schemas.AuthorizationVerificationInput(
        resource="/some-resource", action=mlrun.api.schemas.AuthorizationAction.create
    )

    async def _mock_successful_query_permissions(resource, action, *args):
        assert authorization_verification_input.resource == resource
        assert authorization_verification_input.action == action

    mlrun.api.utils.auth.verifier.AuthVerifier().query_permissions = (
        _mock_successful_query_permissions
    )
    response = client.post(
        "authorization/verifications", json=authorization_verification_input.dict()
    )
    assert response.status_code == http.HTTPStatus.OK.value
