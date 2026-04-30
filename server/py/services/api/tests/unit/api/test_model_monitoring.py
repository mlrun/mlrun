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

import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.nuclio.function
import mlrun.utils.helpers


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


class TestUpdateControllerAuthToken:
    """Tests for ML-12021: Preserve auth token in update_model_monitoring_controller"""

    TOKEN_NAME = "my-iguazio-token"

    def test_auth_token_round_trips_through_nuclio_spec(self):
        spec = mlrun.runtimes.nuclio.function.NuclioSpec()
        mlrun.utils.helpers.set_auth_token_name(spec, self.TOKEN_NAME)

        spec_dict = spec.to_dict()
        extracted = spec_dict.get("auth", {}).get("token_name")

        assert extracted == self.TOKEN_NAME

    def test_nuclio_spec_without_auth_extracts_none(self):
        spec = mlrun.runtimes.nuclio.function.NuclioSpec()

        spec_dict = spec.to_dict()
        extracted = spec_dict.get("auth", {}).get("token_name")

        assert extracted is None

    @patch(
        "services.api.api.endpoints.model_monitoring.process_model_monitoring_secret"
    )
    def test_common_params_propagates_token_to_monitoring_deployment(
        self, _mock_secret
    ):
        from services.api.api.endpoints.model_monitoring import _CommonParams

        commons = _CommonParams(
            project="test-project",
            auth_info=mlrun.common.schemas.AuthInfo(),
            db_session=Mock(),
            auth_token_name=self.TOKEN_NAME,
        )

        deployment = commons.get_monitoring_deployment()

        assert deployment._auth_token_name == self.TOKEN_NAME


class TestGetModelMonitoringURL:
    """Tests for the GET /projects/{project}/model-monitoring/stream-pod-http-url endpoint."""

    _PROJECT = "test-mm-url"
    _URL_PATH = f"projects/{_PROJECT}/model-monitoring/stream-pod-http-url"

    @pytest.fixture(autouse=True)
    def _bypass_auth(self):
        with patch("framework.api.deps.authenticate_request"):
            with patch(
                "framework.utils.auth.verifier.AuthVerifier.query_project_permissions"
            ):
                yield

    @pytest.fixture
    def mock_get_function(self):
        with patch("services.api.crud.Functions.get_function") as mock:
            yield mock

    def test_function_not_found_returns_404(self, client, mock_get_function):
        mock_get_function.return_value = None
        resp = client.get(self._URL_PATH)
        assert resp.status_code == HTTPStatus.NOT_FOUND, resp.text

    def test_function_not_ready_returns_412(self, client, mock_get_function):
        mock_get_function.return_value = {"status": {"state": "deploying"}}
        resp = client.get(self._URL_PATH)
        assert resp.status_code == HTTPStatus.PRECONDITION_FAILED, resp.text

    def test_returns_internal_url(self, client, mock_get_function):
        # Even when external_invocation_urls is populated, always use the internal URL.
        mock_get_function.return_value = {
            "status": {
                "state": "ready",
                "external_invocation_urls": ["external-host:8080/path"],
                "internal_invocation_urls": ["internal:8080"],
            }
        }
        resp = client.get(self._URL_PATH)
        assert resp.status_code == HTTPStatus.OK, resp.text
        assert resp.json() == "http://internal:8080"

    def test_returns_none_when_no_internal_url(self, client, mock_get_function):
        # No internal_invocation_urls → return None (external URL is not used).
        mock_get_function.return_value = {
            "status": {
                "state": "ready",
                "external_invocation_urls": ["external-host:8080/path"],
                "internal_invocation_urls": [],
            }
        }
        resp = client.get(self._URL_PATH)
        assert resp.status_code == HTTPStatus.OK, resp.text
        assert resp.json() is None
