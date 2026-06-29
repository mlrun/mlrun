# Copyright 2026 Iguazio
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

"""Handlers that return mlrun Response wrappers — for ML-12706 system tests."""

import mlrun.errors
from mlrun.serving.server import Response


def error_response_handler(body, **kwargs):
    """Return Response(status_code=404) — result_handler must skip output mapping."""
    return Response(
        body={
            "error": {
                "message": "Response with id resp_x not found",
                "type": "invalid_request_error",
            }
        },
        status_code=404,
        content_type="application/json",
    )


def success_response_handler(body, **kwargs):
    """Return Response(status_code=200) — output mapping must apply, status code preserved."""
    return Response(
        body={"id": "resp_1", "object": "response", "extra_field": "filter"},
        status_code=200,
        content_type="application/json",
    )


def async_dispatcher_handler(body, mlrun_request_path, **kwargs):
    """Dispatch based on request path — for ML-12777 async system test.

    Used inside a single deploy to exercise multiple async post-processing
    paths from one test (avoids deploying per scenario).
    """
    if mlrun_request_path == "/missing_mandatory":
        # Dict missing the mandatory $.id mapping → result_handler should raise 422.
        return {"no_id": "value"}
    if mlrun_request_path == "/raising":
        # Raised exception in async path → precise 404 (not generic 500).
        raise mlrun.errors.MLRunNotFoundError("resource missing")
    if mlrun_request_path == "/error_response":
        # Explicit Response(404) → mapping skipped (non-2xx), body returned intact.
        return Response(
            body={
                "error": {
                    "message": "Response with id resp_x not found",
                    "type": "invalid_request_error",
                }
            },
            status_code=404,
            content_type="application/json",
        )
    if mlrun_request_path == "/success":
        # Response(200) + dict body matching the success contract → mapping reshapes.
        # input_id / input_object come from input_body_mappings on this endpoint.
        return Response(
            body={
                "input_id": kwargs.get("input_id"),
                "input_object": kwargs.get("input_object"),
                "extra_field": "filter",
            },
            status_code=200,
            content_type="application/json",
        )
    raise mlrun.errors.MLRunBadRequestError(
        f"unknown dispatcher path: {mlrun_request_path}"
    )


def raising_handler(body, **kwargs):
    raise mlrun.errors.MLRunNotFoundError("resource missing")
