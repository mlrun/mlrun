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

"""Nuclio function handler used by TestHTTPIngest system test.

MLRun injects the following env vars at deploy time:
    MODEL_MONITORING_URL   — HTTP URL of the monitoring stream pod.
    MODEL_ENDPOINT_UID     — UID of the primary model endpoint.
    MODEL_ENDPOINTS_MAP    — JSON {name: uid} map when multiple endpoints exist.

When invoked the handler pushes prediction events for every registered
endpoint, demonstrating the USER_EP ingest flow from inside a pod.

Expected request body (JSON):
    {"num_events": <int>}   # optional, defaults to 1
"""

import json
import os

import requests

_MONITORING_URL = os.environ.get("MODEL_MONITORING_URL", "")
_ENDPOINT_UID = os.environ.get("MODEL_ENDPOINT_UID", "")
_ENDPOINT_NAME = os.environ.get("MODEL_ENDPOINT_NAME", "")
# {name: uid} for all endpoints; set only when there are multiple endpoints
_ENDPOINTS_MAP: dict[str, str] = json.loads(os.environ.get("MODEL_ENDPOINTS_MAP", "{}"))

# reverse map {uid: name}; primary endpoint falls back to MODEL_ENDPOINT_NAME
_uid_to_name: dict[str, str] = {uid: name for name, uid in _ENDPOINTS_MAP.items()}
if _ENDPOINT_UID and _ENDPOINT_UID not in _uid_to_name:
    _uid_to_name[_ENDPOINT_UID] = _ENDPOINT_NAME


def _all_endpoint_uids() -> list[str]:
    """Return UIDs for every registered endpoint (deduped, preserving order)."""
    seen = set()
    uids = []
    for uid in [_ENDPOINT_UID] + list(_ENDPOINTS_MAP.values()):
        if uid and uid not in seen:
            seen.add(uid)
            uids.append(uid)
    return uids


def handler(context, event):
    if not _MONITORING_URL:
        return context.Response(
            body="MODEL_MONITORING_URL env var is not set",
            status_code=503,
            content_type="text/plain",
        )
    if not _ENDPOINT_UID:
        return context.Response(
            body="MODEL_ENDPOINT_UID env var is not set",
            status_code=503,
            content_type="text/plain",
        )

    body = event.body
    if isinstance(body, (bytes, bytearray)):
        body = json.loads(body) if body else {}
    body = body or {}
    num_events = int(body.get("num_events", 1))

    monitoring_url = _MONITORING_URL.rstrip("/")

    # Test-only: send one malformed event and return the stream pod status code.
    if body.get("test_bad_payload"):
        bad_resp = requests.post(
            monitoring_url,
            json={
                "model_endpoint_uid": _ENDPOINT_UID,
                "model_endpoint_name": _ENDPOINT_NAME,
                # inputs and outputs intentionally omitted to trigger 400
            },
            timeout=10,
        )
        return context.Response(
            body=json.dumps({"bad_payload_status": bad_resp.status_code}),
            status_code=200,
            content_type="application/json",
        )
    endpoint_uids = _all_endpoint_uids()

    pushed = 0
    for endpoint_id in endpoint_uids:
        for i in range(num_events):
            payload = {
                "model_endpoint_uid": endpoint_id,
                "model_endpoint_name": _uid_to_name.get(endpoint_id, ""),
                "inputs": {
                    "age": float(i),
                    "income": float(i + 1),
                    "credit_score": float(i + 2),
                    "balance": float(i + 3),
                },
                "outputs": {"approved": float(i % 2)},
            }
            resp = requests.post(monitoring_url, json=payload, timeout=10)
            context.logger.info(
                f"Stream pod response for event {i} endpoint {endpoint_id}: "
                f"status={resp.status_code} body={resp.text[:200]}"
            )
            if resp.status_code != 202:
                context.logger.warn(
                    f"Event {i} for {endpoint_id} rejected: "
                    f"{resp.status_code} {resp.text}"
                )
            else:
                pushed += 1

    context.logger.info(
        f"Pushed {pushed} events across {len(endpoint_uids)} endpoint(s)"
    )
    return context.Response(
        body=json.dumps({"pushed": pushed, "endpoints": len(endpoint_uids)}),
        status_code=200,
        content_type="application/json",
    )
