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

import unittest.mock
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

import mlrun.common.schemas

API_USER_SECRETS_PATH = "/user-secrets/tokens"


@pytest.mark.parametrize(
    "user_id,username",
    [
        (None, None),  # both missing
        (None, "some-username"),  # user_id missing
        ("some-id", None),  # username missing
    ],
)
def test_store_secret_tokens_missing_authentication_details(
    client: TestClient, user_id, username
) -> None:
    secret_tokens_data = [{"name": "some-token", "token": "offline.jwt"}]

    # Patch the dependency that provides auth_info to simulate missing user_id and username
    with unittest.mock.patch(
        "framework.api.deps.authenticate_request"
    ) as mock_authenticate:
        mock_authenticate.return_value = mlrun.common.schemas.AuthInfo(
            user_id=user_id,
            username=username,
        )

        response = client.put(
            API_USER_SECRETS_PATH,
            json=secret_tokens_data,
        )

    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
