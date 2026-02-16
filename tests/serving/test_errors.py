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
#
"""Tests for error handling and HTTP status codes in serving runtime"""

from http import HTTPMethod
from typing import cast

import pytest

import mlrun
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig, ServingRuntime


# Helper functions for error status code tests
def raise_404_error(event):
    """Helper function that raises MLRunNotFoundError"""
    raise mlrun.errors.MLRunNotFoundError("Resource not found")


def raise_400_error(event):
    """Helper function that raises MLRunBadRequestError"""
    raise mlrun.errors.MLRunBadRequestError("Invalid request")


def raise_403_error(event):
    """Helper function that raises MLRunAccessDeniedError"""
    raise mlrun.errors.MLRunAccessDeniedError("Access denied")


def raise_409_error(event):
    """Helper function that raises MLRunConflictError"""
    raise mlrun.errors.MLRunConflictError("Resource conflict")


def raise_500_error(event):
    """Helper function that raises MLRunInternalServerError"""
    raise mlrun.errors.MLRunInternalServerError("Internal server error")


def raise_value_error(event):
    """Helper function that raises ValueError"""
    raise ValueError("Some generic error")


def raise_runtime_error(event):
    """Helper function that raises RuntimeError"""
    raise RuntimeError("Runtime error occurred")


@pytest.mark.parametrize(
    "error_handler,expected_status_code,error_class_name,error_message",
    [
        # MLRun exceptions with specific status codes
        (
            "raise_404_error",
            404,
            "MLRunNotFoundError",
            "Resource not found",
        ),
        (
            "raise_400_error",
            400,
            "MLRunBadRequestError",
            "Invalid request",
        ),
        (
            "raise_403_error",
            403,
            "MLRunAccessDeniedError",
            "Access denied",
        ),
        (
            "raise_409_error",
            409,
            "MLRunConflictError",
            "Resource conflict",
        ),
        (
            "raise_500_error",
            500,
            "MLRunInternalServerError",
            "Internal server error",
        ),
        # Non-MLRun exceptions (backwards compatibility: should return 400)
        (
            "raise_value_error",
            400,
            "ValueError",
            "Some generic error",
        ),
        (
            "raise_runtime_error",
            400,
            "RuntimeError",
            "Runtime error occurred",
        ),
    ],
    ids=[
        "404_not_found",
        "400_bad_request",
        "403_access_denied",
        "409_conflict",
        "500_internal_server_error",
        "400_value_error_backwards_compat",
        "400_runtime_error_backwards_compat",
    ],
)
def test_error_status_codes(
    error_handler: str,
    expected_status_code: int,
    error_class_name: str,
    error_message: str,
) -> None:
    """Test that different error types return proper HTTP status codes

    This test verifies that:
    - MLRun errors return their specific status codes (404, 403, 409, 500, etc.)
    - Non-MLRun exceptions return 400 for backwards compatibility
    - Error messages are properly included in response body
    """
    fn = cast(ServingRuntime, mlrun.new_function("test-error", kind="serving"))
    graph = fn.set_topology("flow", engine="sync")
    graph.to(
        name="error_step",
        handler=error_handler,
    ).respond()

    server = fn.to_mock_server()
    try:
        resp = server.test("/", method="GET", body="test", silent=True)
        assert resp.status_code == expected_status_code, (
            f"Expected status code {expected_status_code} for {error_class_name}, "
            f"got {resp.status_code}"
        )
        assert error_class_name in resp.body, (
            f"Expected error class '{error_class_name}' in response body, "
            f"got: {resp.body}"
        )
        assert error_message in resp.body, (
            f"Expected error message '{error_message}' in response body, "
            f"got: {resp.body}"
        )
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize(
    "endpoint_path,expected_status_code,error_class_name,error_pattern",
    [
        (
            "/api/v1/not-found",
            404,
            "MLRunNotFoundError",
            "Endpoint not found: GET /api/v1/not-found",
        ),
        (
            "/api/v1/forbidden",
            400,
            "MLRunBadRequestError",
            "Access forbidden to GET /api/v1/forbidden",
        ),
    ],
    ids=["404_endpoint_not_found", "400_endpoint_forbidden"],
)
def test_api_handler_error_status_codes(
    endpoint_path: str,
    expected_status_code: int,
    error_class_name: str,
    error_pattern: str,
) -> None:
    """Test that API handler returns correct status codes for different error scenarios

    This test verifies that:
    - Non-existent endpoints return 404
    - Forbidden endpoints return 400 (current behavior)
    """
    fn = cast(ServingRuntime, mlrun.new_function("test-api-handler", kind="serving"))

    config = APIHandlerConfig()
    # Add only one allowed endpoint - others will be not found
    config.add_endpoint_handler(
        "/api/v1/exists", HTTPMethod.GET, APIHandlerAction.ALLOW
    )

    # Add a forbidden endpoint if we're testing that case
    if "forbidden" in endpoint_path:
        config.add_endpoint_handler(
            endpoint_path, HTTPMethod.GET, APIHandlerAction.FORBID
        )

    fn.set_api_handler_config(config)
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="echo", handler="(event)").respond()

    server = fn.to_mock_server()
    try:
        resp = server.test(endpoint_path, method="GET", body="test", silent=True)
        assert resp.status_code == expected_status_code, (
            f"Expected status code {expected_status_code} for {endpoint_path}, "
            f"got {resp.status_code}"
        )
        assert error_class_name in resp.body, (
            f"Expected error class '{error_class_name}' in response body, "
            f"got: {resp.body}"
        )
        assert error_pattern in resp.body, (
            f"Expected error pattern '{error_pattern}' in response body, "
            f"got: {resp.body}"
        )
    finally:
        server.wait_for_completion()
