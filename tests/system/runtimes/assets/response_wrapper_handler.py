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
