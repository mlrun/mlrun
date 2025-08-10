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

from http import HTTPStatus
from typing import Any

import pytest
from fastapi.testclient import TestClient

TEST_PROJECT = "test-model-endpoints"


@pytest.mark.parametrize(
    ("params", "expected_status"),
    [
        ({}, HTTPStatus.UNPROCESSABLE_ENTITY),
        ({"application-name": "app1"}, HTTPStatus.NO_CONTENT),
    ],
)
def test_delete_model_endpoint(
    client: TestClient, params: dict[str, Any], expected_status: HTTPStatus
) -> None:
    resp = client.delete(
        f"projects/{TEST_PROJECT}/model-monitoring-metrics",
        params=params,
    )
    assert resp.status_code == expected_status, resp.text
