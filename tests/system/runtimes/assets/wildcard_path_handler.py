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


def handle_wildcard(
    body,
    mlrun_request_path: str | None = None,
    mlrun_request_method: str | None = None,
    **kwargs,
) -> dict:
    """Handler that receives mlrun_request_path and mlrun_request_method injected by the API handler.

    Used by system tests to verify that:
    - wildcard ``*`` patterns route requests to the correct handler
    - ``include_url_info=True`` injects both ``mlrun_request_path`` (normalized
      request path, no query string) and ``mlrun_request_method`` (HTTP method)

    Args:
        mlrun_request_path:   Injected by APIHandler when include_url_info=True.
        mlrun_request_method: Injected by APIHandler when include_url_info=True.
        **kwargs:             Any additional parameters from query string / body_map.

    Returns:
        Dict with matched_path, matched_method, and any extra params for verification.
    """
    return {
        "matched_path": mlrun_request_path,
        "matched_method": mlrun_request_method,
        "extra": kwargs,
    }
