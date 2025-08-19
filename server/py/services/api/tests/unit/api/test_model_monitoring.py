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

from collections.abc import Iterator
from http import HTTPStatus
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_delete_application_records() -> Iterator[Mock]:
    with patch("framework.api.deps.authenticate_request"):
        with patch("services.api.api.endpoints.model_monitoring._verify_authorization"):
            with patch(
                "services.api.api.endpoints.model_monitoring.process_model_monitoring_secret"
            ):
                with patch(
                    "services.api.crud.model_monitoring.deployment.MonitoringDeployment.delete_application_records"
                ) as mock:
                    yield mock


@pytest.mark.parametrize(
    ("params", "expected_status"),
    [
        ({}, HTTPStatus.UNPROCESSABLE_ENTITY),
        (
            {"application-name": "app1"},
            HTTPStatus.NO_CONTENT,
        ),
        (
            {
                "application-name": "app2",
                "endpoint-id": [
                    "0ca48b46b0de460599faf6bead10dbe8",
                    "1ab23c45d6ef7890123456789abcdef0",
                ],
            },
            HTTPStatus.NO_CONTENT,
        ),
    ],
)
def test_delete_model_monitoring_metrics(
    mock_delete_application_records: Mock,
    client: TestClient,
    params: dict[str, Any],
    expected_status: HTTPStatus,
) -> None:
    resp = client.delete(
        "projects/test-model-monitoring/model-monitoring/metrics",
        params=params,
        headers={"x-mlrun-client-version": "1.10.0"},
    )
    assert resp.status_code == expected_status, resp.text
    if expected_status == HTTPStatus.NO_CONTENT and params:
        mock_delete_application_records.assert_called_once_with(
            application_name=params.get("application-name"),
            endpoint_ids=params.get("endpoint-id"),
        )
