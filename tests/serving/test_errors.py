# Copyright 2026 Iguazio
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
from typing import Callable, cast

import pytest

import mlrun
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig, ServingRuntime


def _make_error_handler(error_class: type[Exception], message: str) -> Callable:
    """Factory function to create error handler functions

    Args:
        error_class: The exception class to raise
        message: The error message to include

    Returns:
        A handler function that raises the specified exception
    """

    def handler(event):
        raise error_class(message)

    handler.__name__ = f"raise_{error_class.__name__}"
    return handler


@pytest.mark.parametrize(
    "error_class,error_message,expected_status_code",
    [
        # MLRun exceptions with specific status codes
        (mlrun.errors.MLRunNotFoundError, "Resource not found", 404),
        (mlrun.errors.MLRunBadRequestError, "Invalid request", 400),
        (mlrun.errors.MLRunAccessDeniedError, "Access denied", 403),
        (mlrun.errors.MLRunConflictError, "Resource conflict", 409),
        (mlrun.errors.MLRunInternalServerError, "Internal server error", 500),
        (mlrun.errors.MLRunMethodNotAllowedError, "Method not allowed", 405),
        (mlrun.errors.MLRunUnprocessableEntityError, "Unprocessable entity", 422),
        # Non-MLRun exceptions (backwards compatibility: should return 400)
        (ValueError, "Some generic error", 400),
        (RuntimeError, "Runtime error occurred", 400),
    ],
    ids=[
        "404_not_found",
        "400_bad_request",
        "403_access_denied",
        "409_conflict",
        "500_internal_server_error",
        "405_method_not_allowed",
        "422_unprocessable_entity",
        "400_value_error_backwards_compat",
        "400_runtime_error_backwards_compat",
    ],
)
def test_error_status_codes(
    error_class: type[Exception],
    error_message: str,
    expected_status_code: int,
) -> None:
    """Test that different error types return proper HTTP status codes

    This test verifies that:
    - MLRun errors return their specific status codes (404, 403, 409, 500, etc.)
    - Non-MLRun exceptions return 400 for backwards compatibility
    - Error messages are properly included in response body
    """
    # Create error handler dynamically
    error_handler = _make_error_handler(error_class, error_message)

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
            f"Expected status code {expected_status_code} for {error_class.__name__}, "
            f"got {resp.status_code}"
        )
        assert error_class.__name__ in resp.body, (
            f"Expected error class '{error_class.__name__}' in response body, "
            f"got: {resp.body}"
        )
        assert error_message in resp.body, (
            f"Expected error message '{error_message}' in response body, "
            f"got: {resp.body}"
        )
    finally:
        server.wait_for_completion()


@pytest.mark.parametrize(
    "endpoint_path,expected_status_code,error_class_name,error_pattern,endpoints_config",
    [
        (
            "/api/v1/not-found",
            404,
            "MLRunNotFoundError",
            "Endpoint not found: GET /api/v1/not-found",
            {"/api/v1/exists": APIHandlerAction.ALLOW},
        ),
        (
            "/api/v1/forbidden",
            403,
            "MLRunAccessDeniedError",
            "Access forbidden to GET /api/v1/forbidden",
            {"/api/v1/forbidden": APIHandlerAction.FORBID},
        ),
        (
            "/api/v1/resource",
            405,
            "MLRunMethodNotAllowedError",
            "Method not allowed: GET /api/v1/resource",
            {"POST:/api/v1/resource": APIHandlerAction.ALLOW},
        ),
    ],
    ids=[
        "404_endpoint_not_found",
        "403_endpoint_forbidden",
        "405_method_not_allowed",
    ],
)
def test_api_handler_error_status_codes(
    endpoint_path: str,
    expected_status_code: int,
    error_class_name: str,
    error_pattern: str,
    endpoints_config: dict,
) -> None:
    """Test that API handler returns correct status codes for different error scenarios

    This test verifies that:
    - Non-existent endpoints return 404
    - Forbidden endpoints return 403
    - Wrong HTTP method for existing endpoint returns 405
    """
    fn = cast(ServingRuntime, mlrun.new_function("test-api-handler", kind="serving"))

    config = APIHandlerConfig()
    # Add configured endpoints
    for endpoint_key, action in endpoints_config.items():
        # Parse method and path from endpoint_key
        if ":" in endpoint_key:
            method_str, path = endpoint_key.split(":", 1)
            method = HTTPMethod[method_str]
        else:
            method = HTTPMethod.GET
            path = endpoint_key

        config.add_endpoint_handler(path, method, action)

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
