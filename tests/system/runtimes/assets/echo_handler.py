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


import json
from http import HTTPStatus

import nuclio_sdk


def handler(
    context: nuclio_sdk.Context, event: nuclio_sdk.Event
) -> nuclio_sdk.Response:
    """Plain Nuclio handler that echoes the request body as JSON.

    Used by the ML-12228 regression test to verify that invoking any
    Nuclio function with method="HEAD" raises JSONDecodeError in
    RemoteRuntime.invoke() – independent of the API handler feature.
    """
    body = event.body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            body = body

    return context.Response(
        body=json.dumps({"echo": body}),
        headers={},
        content_type="application/json",
        status_code=HTTPStatus.OK,
    )
